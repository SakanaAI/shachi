import copy
import json
import logging
import random
from collections import defaultdict
from collections.abc import AsyncIterator, Mapping, Sequence
from enum import Enum, auto
from typing import Any, Literal, TypeAlias, cast

import inflect
import pydantic
import trueskill

from shachi.base import Environment, Observation, Task

from .auction_item import AuctionItem, create_items
from .bidder import DESIRE_DESC, Bidder, BiddersStatus
from .bidder_conventions import get_bidder_name
from .observation import (
    AuctionMessage,
    BiddingObservation,
    BidResponse,
    PlanObservation,
    PlanResponse,
    RePlanObservation,
    ReplanResponse,
    SummarizeObservation,
    SummarizeResponse,
)
from .prompt_base import SYSTEM_MESSAGE

p = inflect.engine()

# Default directory for storing auction logs (can be configured)
LOG_DIR = "logs"

logger = logging.getLogger(__name__)


ItemOrder: TypeAlias = Literal["random", "desc", "asc"]


class AuctionStage(Enum):
    PLAN = auto()
    BID_COLLECT = auto()
    BID_PROCESS = auto()
    HAMMER_CHECK = auto()
    SUMMARIZE = auto()
    REPLAN = auto()
    NEXT_ITEM = auto()
    END = auto()


class PassedItemResult(pydantic.BaseModel):
    item_id: int = pydantic.Field(..., description="The ID of the auction item.")
    true_value: int = pydantic.Field(..., description="The true value of the auction item.")


class WonItemResult(pydantic.BaseModel):
    item_id: int = pydantic.Field(..., description="The ID of the auction item.")
    true_value: int = pydantic.Field(..., description="The true value of the auction item.")

    bidder_id: int = pydantic.Field(..., description="The ID of the bidder who won this item.")
    bid_amount: int = pydantic.Field(..., description="The bidding price for this item.")

    @property
    def profit(self) -> int:
        return self.true_value - self.bid_amount


class AuctionResult(pydantic.BaseModel):
    item_results: list[PassedItemResult | WonItemResult] = pydantic.Field(
        ..., description="The auction result for each item."
    )
    bidders: list[Bidder] = pydantic.Field(default_factory=list, description="Bidders who participated in the auction.")

    def calculate_result(self) -> dict:
        result = dict()
        for bidder in self.bidders:
            result[bidder.id] = {"profit": 0, "won_items": 0}

        for item_result in self.item_results:
            if not isinstance(item_result, WonItemResult):
                continue
            result[item_result.bidder_id]["profit"] += item_result.profit
            result[item_result.bidder_id]["won_items"] += 1
        return result


class AuctionEnvironment(Environment):
    """
    Environment for running auctions following the shachi.Environment interface.

    Manages the staged flow of an auction (Plan, Bid, Summarize, Replan)
    and interacts with agents via Observations and Responses.
    """

    def __init__(
        self,
        items: list[dict],
        bidders: list[dict[str, Any]],
        min_markup_pct: float = 0.1,
        enable_discount: bool = False,
        discount_percentage: float = 0.5,
        max_discount_rounds: int = 3,
        item_order: ItemOrder = "random",
    ):
        self._all_items = items
        self._initial_bidder_configs = bidders
        self._num_agents = sum(map(lambda bidder: 1 if not bidder.get("is_rule_based", False) else 0, bidders))
        self._min_markup_pct = min_markup_pct
        self._enable_discount = enable_discount
        self._discount_percentage = discount_percentage
        self._max_discount_rounds = max_discount_rounds
        self._item_order = item_order

        self._items: list[AuctionItem] = []
        self._bidders: dict[int, Bidder] = {}  # agent_id -> Bidder state
        self._current_stage: AuctionStage = AuctionStage.PLAN
        self._current_item_index: int = -1
        self._current_bid_round: int = 0
        self._highest_bid: int = -1
        self._highest_bidder_id: int | None = None
        self._bids_this_round: dict[int, BidResponse] = {}  # Store validated bids
        self._raw_responses_this_step: dict[int, Any] = {}  # Store raw agent responses
        self._bidding_history_item: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._auction_logs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._discount_rounds_applied: int = 0
        self._item_failed_to_sell_round: bool = False  # Flag if item got no bids in a round
        self._time_step: int = 0

        self._rebid_counts: int = 0

        self._auction_result = AuctionResult(item_results=[], bidders=list(self._bidders.values()))

        self.prev_round_max_bid: int = -1

    def get_result(self) -> AuctionResult:
        return self._auction_result

    def num_agents(self) -> int:
        return self._num_agents

    def get_default_agent_configs(self) -> list[dict]:
        for agent_id, bidder_config in enumerate(self._initial_bidder_configs):
            bidder_config["system_prompt"] = SYSTEM_MESSAGE.format(
                bidder_name=get_bidder_name(agent_id),
                desire_desc=DESIRE_DESC[bidder_config["desire_desc"]],
            )

        return self._initial_bidder_configs

    def done(self) -> bool:
        is_done = self._current_stage == AuctionStage.END
        if is_done:
            logger.info("Auction Done! Auction Summary:")
            result_str = json.dumps(self._auction_result.calculate_result(), indent=4)
            logger.info(result_str)
        return is_done

    async def reset(self) -> dict[int, Observation]:
        """Resets the environment to its initial state for a new auction."""
        self._time_step = 0
        self._items = create_items(self._all_items, item_order=self._item_order)

        self._bidders = {}
        bidder_status_init = {}
        for i, config in enumerate(self._initial_bidder_configs):
            bidder_id = i
            bidder_state = Bidder.create(id=bidder_id, **config)
            bidder_state.reset_for_new_auction(self._items)
            self._bidders[bidder_id] = bidder_state
            bidder_status_init[bidder_id] = {
                "profit": 0,
                "items_won": [],
            }

        # Initialize bidder status quo and cache
        for bidder_id, bidder in self._bidders.items():
            bidder.status_quo.bidders_status = [
                BiddersStatus(bidder_name=b.name, profit=0, winning_bids=[]) for b in self._bidders.values()
            ]

        self._current_stage = AuctionStage.PLAN
        self._current_item_index = -1
        self._current_bid_round = 0
        self._highest_bid = -1
        self._highest_bidder_id = None
        self._bids_this_round = {}
        self._raw_responses_this_step = {}
        self._bidding_history_item = defaultdict(list)
        self._auction_logs = defaultdict(list)
        self._auction_result = AuctionResult(item_results=[], bidders=list(self._bidders.values()))
        self.prev_round_max_bid = -1

        logger.info("--- Auction Reset ---")
        logger.info(f"Items ({len(self._items)}): {[item.name for item in self._items]}")
        logger.info(f"Bidders ({self._num_agents}): {[b.name for b in self._bidders.values()]}")

        logger.info(f"\n--- Time Step {self._time_step}, Stage: {self._current_stage.name} ---")
        observations = self._get_plan_observations()
        self._current_stage = AuctionStage.NEXT_ITEM
        return cast(dict[int, Observation], observations)

    async def step(self, responses: dict[int, str | pydantic.BaseModel | None]) -> dict[int, Observation]:
        """Advances the simulation by one time step based on agent responses."""
        self._time_step += 1
        self._raw_responses_this_step = responses
        logger.info(f"\n--- Time Step {self._time_step}, Stage: {self._current_stage.name} ---")
        logger.debug(f"Raw Responses: {responses}")

        next_observations: Mapping[int, Observation] = {}

        try:
            if self._current_stage == AuctionStage.PLAN:
                # Process PlanResponses (store plans in Bidder state)
                self._process_plan_responses(responses)
                self._current_stage = AuctionStage.NEXT_ITEM
                # No observation needed immediately, proceed to NEXT_ITEM logic

            if self._current_stage == AuctionStage.NEXT_ITEM:
                proceed = self._prepare_next_item()
                if proceed:
                    self._current_stage = AuctionStage.BID_COLLECT
                    next_observations = self._get_bidding_observations()
                else:
                    self._current_stage = AuctionStage.END
                    # No observations needed for END state

            elif self._current_stage == AuctionStage.BID_COLLECT:
                # Process BidResponses
                valid_bids, invalid_bids = self._validate_and_collect_bids(responses)
                if self._rebid_counts > 0:
                    # This is a re-bid pass: only the bidders whose bids were rejected are
                    # asked again, so `valid_bids` covers just them. Merge instead of
                    # replacing, otherwise bids already accepted earlier in this same round
                    # would be silently dropped and the item could be hammered to the wrong
                    # bidder at a lower price.
                    self._bids_this_round.update(valid_bids)
                else:
                    self._bids_this_round = valid_bids
                # Handle invalid bids only up to once
                if self._rebid_counts == 0 and invalid_bids:
                    logger.warning(f"WARNING: Invalid bids received: {invalid_bids}")
                    self._rebid_counts += 1

                    self._current_stage = AuctionStage.BID_COLLECT
                    next_observations = self._get_bidding_observations(invalid_bids_to_rebid=invalid_bids)
                else:
                    self._current_stage = AuctionStage.BID_PROCESS

            elif self._current_stage == AuctionStage.BID_PROCESS:
                self._rebid_counts = 0

                self._process_bids_this_round()
                self._current_stage = AuctionStage.HAMMER_CHECK

            elif self._current_stage == AuctionStage.HAMMER_CHECK:
                is_sold = self._check_hammer()
                if is_sold:
                    logger.info(f"Item '{self._get_current_item().name}' Sold/Passed.")
                    self._record_sale_or_pass()
                    next_observations = self._get_summarize_observations()
                    self._current_stage = AuctionStage.SUMMARIZE
                elif (
                    self._item_failed_to_sell_round
                    and self._enable_discount
                    and self._discount_rounds_applied < self._max_discount_rounds
                ):
                    # Apply discount and restart bidding round for this item
                    logger.info(f"Item '{self._get_current_item().name}' failed to sell, applying discount.")
                    self._get_current_item().lower_price(self._discount_percentage)
                    self._discount_rounds_applied += 1
                    self._reset_bidding_round_state(keep_history=True)
                    # Allow withdrawn bidders back in
                    for bidder in self._bidders.values():
                        bidder.withdraw = False
                    next_observations = self._get_bidding_observations()
                    self._current_stage = AuctionStage.BID_COLLECT
                else:
                    # Continue to next bidding round for the same item
                    self._current_bid_round += 1
                    next_observations = self._get_bidding_observations()
                    self._current_stage = AuctionStage.BID_COLLECT

            elif self._current_stage == AuctionStage.SUMMARIZE:
                if self._is_summarize_response(responses):
                    self._process_summarize_responses(cast(dict[int, SummarizeResponse], responses))
                else:
                    raise RuntimeError(
                        f"Internal Error, responses are expected to be of type Dict[int, SummarizeResponse], "
                        f"but got type {type(responses)}; {responses}"
                    )
                # Check if replanning is needed/enabled
                if self._current_item_index < len(self._items) - 1:
                    needs_replan = any(not bidder.is_rule_based for bidder in self._bidders.values())
                    if needs_replan:
                        next_observations = self._get_replan_observations()
                        self._current_stage = AuctionStage.REPLAN
                    else:
                        self._finalize_item_auction()
                        self._current_stage = AuctionStage.NEXT_ITEM
                        # No observation needed, proceed to NEXT_ITEM logic in next step
                else:
                    # No more items, finalize and end
                    self._finalize_item_auction()
                    self._current_stage = AuctionStage.END

            elif self._current_stage == AuctionStage.REPLAN:
                self._process_replan_responses(responses)
                self._finalize_item_auction()
                self._current_stage = AuctionStage.NEXT_ITEM
                # No observation needed, proceed to NEXT_ITEM logic in next step

            elif self._current_stage == AuctionStage.END:
                raise RuntimeError("Please do not call `step` after the auction ended, please call reset to restart.")

        except Exception as e:
            logger.error(f"Error during step {self._time_step}, stage {self._current_stage.name}: {e}")
            self._current_stage = AuctionStage.END
            raise e

        if not next_observations and self._current_stage != AuctionStage.END:
            if self._current_stage == AuctionStage.NEXT_ITEM:
                next_observations = dict()
            elif self._current_stage == AuctionStage.BID_COLLECT:
                next_observations = self._get_bidding_observations()

        final_observations = {
            bidder_id: next_observations[bidder_id] for bidder_id in self._bidders if bidder_id in next_observations
        }

        logger.debug(f"Next Stage: {self._current_stage.name}")
        logger.debug(f"Sending Observations: {final_observations}")
        return final_observations

    @property
    def current_item(self) -> AuctionItem | None:
        if 0 <= self._current_item_index < len(self._items):
            return self._items[self._current_item_index]
        return None

    def _get_plan_observations(self) -> dict[int, PlanObservation]:
        observations = {}
        for bidder_id, bidder in self._bidders.items():
            if bidder.is_rule_based:
                continue

            initial_message = AuctionMessage(
                time=self._time_step,
                src_agent_id=None,
                dst_agent_id=bidder_id,
                content="Welcome to the auction. Please prepare your bidding plan.",
            )
            observations[bidder_id] = PlanObservation(
                agent_id=bidder_id,
                messages=[initial_message],
                budget=bidder.budget,
                items=self._items,
                desire_desc=bidder.desire_desc,
                response_type=PlanResponse,
            )
        return observations

    def _get_bidding_observations(
        self, invalid_bids_to_rebid: dict[int, tuple[str, int]] | None = None
    ) -> dict[int, BiddingObservation]:
        observations = {}

        for bidder_id, bidder in self._bidders.items():
            if bidder.is_rule_based:
                continue

            if invalid_bids_to_rebid is not None and bidder_id not in invalid_bids_to_rebid:
                continue

            if invalid_bids_to_rebid is None:
                auctioneer_msg_content = self._get_auctioneer_bid_request_msg()
            else:
                failed_msg, bid_amount = invalid_bids_to_rebid[bidder_id]
                auctioneer_msg_content = self._get_rebid_msg(fail_msg=failed_msg, bid_price=bid_amount)

            # Filter observation for bidders who haven't withdrawn or bid highest price
            if not (bidder.withdraw or bidder.id == self._highest_bidder_id):
                message = AuctionMessage(
                    time=self._time_step,
                    src_agent_id=None,
                    dst_agent_id=bidder_id,
                    content=auctioneer_msg_content.replace(bidder.name, f"You ({bidder.name})"),
                )
                observations[bidder_id] = BiddingObservation(
                    agent_id=bidder_id,
                    messages=[message],
                    cur_item=self._get_current_item(),
                    auctioneer_msg=message.content,
                    desire_desc=bidder.desire_desc,
                    response_type=BidResponse,
                )

        logger.debug(f"BIDDING OBS: {observations}, HIGHEST_BID: {self._highest_bidder_id}")
        return observations

    def _get_summarize_observations(self) -> dict[int, SummarizeObservation]:
        observations = {}
        bidding_history_text = self._get_all_bidding_history_text()
        hammer_msg = self._get_hammer_msg()

        for bidder_id, bidder in self._bidders.items():
            if bidder.is_rule_based:
                continue

            current_item = self._get_current_item()
            win_lose_msg = (
                f"Congratulations! You won {current_item.name} at ${self._highest_bid}."
                if bidder_id == self._highest_bidder_id
                else f"You lost {current_item.name}."
            )
            prev_status_text = bidder._status_json_to_text(bidder.status_quo.model_dump())

            message = AuctionMessage(
                time=self._time_step,
                src_agent_id=None,
                dst_agent_id=bidder_id,
                content=f"Bidding for {current_item.name} has concluded. {hammer_msg}",
            )

            observations[bidder_id] = SummarizeObservation(
                agent_id=bidder_id,
                messages=[message],
                cur_item=current_item,
                bidding_history_text=bidding_history_text,
                hammer_msg=hammer_msg,
                win_lose_msg=win_lose_msg,
                prev_status_text=prev_status_text,
                response_type=SummarizeResponse,
            )
        return observations

    def _get_replan_observations(
        self,
    ) -> dict[int, RePlanObservation]:
        observations = {}
        remaining_items = self._items[self._current_item_index + 1 :]

        for bidder_id, bidder in self._bidders.items():
            needs_replan = not bidder.is_rule_based
            if needs_replan:
                message = AuctionMessage(
                    time=self._time_step,
                    src_agent_id=None,
                    dst_agent_id=bidder_id,
                    content=(
                        f"Item {self._get_current_item().name} auction finished. "
                        f"Consider replanning for remaining items."
                    ),
                )
                observations[bidder_id] = RePlanObservation(
                    agent_id=bidder_id,
                    messages=[message],
                    status_quo=bidder.status_quo,
                    remaining_items=remaining_items,
                    desire_desc=bidder.desire_desc,
                    response_type=ReplanResponse,
                )

        return observations

    def _process_plan_responses(self, responses: dict[int, Any]) -> None:
        logger.info("Processing Plans...")
        for bidder_id, response in responses.items():
            bidder = self._bidders.get(bidder_id)
            if bidder and isinstance(response, PlanResponse):
                bidder.cur_plan = response.plan
                bidder.plan_history.append(response.plan)
                logger.info(f"  Bidder {bidder_id} Plan: {response.plan[:100]}...")
            elif bidder:
                logger.warning(f"  WARNING: Bidder {bidder_id} provided invalid PlanResponse: {type(response)}")

    def _validate_and_collect_bids(
        self, responses: dict[int, Any]
    ) -> tuple[dict[int, BidResponse], dict[int, tuple[str, int]]]:
        logger.info("Validating & Collecting Bids...")
        valid_bids: dict[int, BidResponse] = {}
        invalid_bids: dict[int, tuple[str, int]] = {}

        for bidder_id, response in responses.items():
            bidder = self._bidders.get(bidder_id)
            if bidder and not bidder.withdraw:
                if isinstance(response, BidResponse):
                    bid_amount = response.bid_amount
                    fail_msg = bidder.bid_sanity_check(bid_amount, self._highest_bid, self._min_markup_pct)
                    if fail_msg is None:
                        valid_bids[bidder_id] = response
                        logger.info(f"  Bidder {bidder_id}: Valid Bid ${bid_amount if bid_amount >= 0 else 'Withdraw'}")
                    else:
                        invalid_bids[bidder_id] = (fail_msg, bid_amount)
                        logger.info(f"  Bidder {bidder_id}: Invalid Bid ({fail_msg}) - Response: {response}")
                elif bidder.is_rule_based:
                    # Rule bid is calculated directly in _process_bids_this_round
                    pass
                elif response is None and bidder_id in self._bidders:
                    logger.info(f"  Bidder {bidder_id}: No response (Treating as Withdraw)")
                    valid_bids[bidder_id] = BidResponse(bid_amount=-1)
                else:
                    invalid_bids[bidder_id] = (
                        f"Invalid response type: {type(response)}",
                        bid_amount,
                    )
                    logger.info(f"  Bidder {bidder_id}: Invalid Response Type ({type(response)})")

        for bidder_id, bidder in self._bidders.items():
            if (
                bidder.is_rule_based
                and not bidder.withdraw
                and bidder_id not in valid_bids
                and bidder_id not in invalid_bids
                and bidder_id != self._highest_bidder_id
            ):
                rule_bid_amount = bidder.bid_rule(self._highest_bid, self._min_markup_pct)
                fail_msg = bidder.bid_sanity_check(rule_bid_amount, self._highest_bid, self._min_markup_pct)
                if fail_msg is None:
                    valid_bids[bidder_id] = BidResponse(bid_amount=rule_bid_amount)
                    logger.info(
                        f"  Bidder {bidder_id} (Rule): Valid Bid ",
                        f"${rule_bid_amount if rule_bid_amount >= 0 else 'Withdraw'}",
                    )
                else:
                    logger.info(f"  Bidder {bidder_id} (Rule): Invalid Bid ({fail_msg}) -> Withdrawing")
                    valid_bids[bidder_id] = BidResponse(bid_amount=-1)

        return valid_bids, invalid_bids

    def _process_bids_this_round(self) -> None:
        logger.info("Processing Bids...")
        current_round_bids = []
        new_highest_bid_this_round = -1
        new_highest_bidder_id_this_round = None
        bidders_in_round = 0

        for bidder_id, bid_response in self._bids_this_round.items():
            bidder = self._bidders[bidder_id]
            bid_amount = bid_response.bid_amount

            bidder.set_withdraw_status(bid_amount)

            bid_info = {"bidder_id": bidder_id, "bid_amount": bid_amount}
            current_round_bids.append(bid_info)

            if bid_amount >= 0:
                bidders_in_round += 1
                if bid_amount > new_highest_bid_this_round:
                    new_highest_bid_this_round = bid_amount
                    new_highest_bidder_id_this_round = bidder_id
                elif bid_amount == new_highest_bid_this_round:
                    # Tie-breaking: randomly choose between current and new bidder
                    if random.choice([True, False]):
                        new_highest_bidder_id_this_round = bidder_id

        self._bidding_history_item[self._current_bid_round].extend(current_round_bids)

        if new_highest_bid_this_round > self._highest_bid:
            self._highest_bid = new_highest_bid_this_round
            self._highest_bidder_id = new_highest_bidder_id_this_round
        elif new_highest_bid_this_round == self._highest_bid and new_highest_bidder_id_this_round is not None:
            self._highest_bidder_id = new_highest_bidder_id_this_round

        logger.info(f"  Round {self._current_bid_round} Bids: {current_round_bids}")
        logger.info(f"  Current Highest Bid for Item: ${self._highest_bid} by Bidder {self._highest_bidder_id}")

        self._item_failed_to_sell_round = bidders_in_round == 0

    def _process_summarize_responses(self, responses: dict[int, SummarizeResponse]) -> None:
        logger.info("Processing Summaries (Updating Beliefs)...")
        for bidder_id, response in responses.items():
            bidder = self._bidders.get(bidder_id)
            assert bidder and isinstance(response, SummarizeResponse)
            logger.debug(f"Proposed status_quo {response.status_quo}")
            bidder.status_quo = response.status_quo

    def _process_replan_responses(self, responses: dict[int, Any]) -> None:
        logger.info("Processing Replans...")
        for bidder_id, response in responses.items():
            bidder = self._bidders.get(bidder_id)
            if bidder and isinstance(response, ReplanResponse):
                bidder.cur_plan = response.plan
                bidder.plan_history.append(response.plan)
                logger.info(f"  Bidder {bidder_id} Plan Updated: {response.plan[:100]}...")
            elif bidder and not bidder.is_rule_based:
                if response is not None:
                    logger.warning(f"  WARNING: Bidder {bidder_id} provided invalid ReplanResponse: {type(response)}")

    def _prepare_next_item(self) -> bool:
        """Moves to the next item in the list. Returns False if no items left."""
        self._current_item_index += 1
        if self.current_item:
            logger.info(
                f"Starting Auction for Item {self._current_item_index + 1}/{len(self._items)}: "
                f"{self.current_item.name} (Price: ${self.current_item.price})"
            )
            self._reset_bidding_round_state()
            for bidder in self._bidders.values():
                bidder.reset_for_new_item()
            return True
        else:
            logger.info("No more items left.")
            return False

    def _reset_bidding_round_state(self, keep_history: bool = False) -> None:
        """Resets state for a new item or when discounting."""
        self._current_bid_round = 0
        self._highest_bid = -1
        self._highest_bidder_id = None
        self._bids_this_round = {}
        self._discount_rounds_applied = 0
        self._item_failed_to_sell_round = False
        if not keep_history:
            self._bidding_history_item = defaultdict(list)

    def _check_hammer(self) -> bool:
        """
        Determines if the hammer should fall based on the last round of bids.
        Condition: Hammer falls if no bids were placed in the last round
                   OR if only one bid was placed in the very first round (round 0).
        """
        bids_in_last_round = self._bidding_history_item[self._current_bid_round]
        logger.info(f"Bids in last round {bids_in_last_round}")
        num_positive_bids = sum(1 for bid in bids_in_last_round if bid["bid_amount"] >= 0)

        highest_bidder_id = self._highest_bidder_id

        if highest_bidder_id is None:
            if num_positive_bids == 0:
                self._fail_to_sell = True
                if self._enable_discount and self._current_bid_round < 3:
                    self._get_current_item().lower_price(0.5)
                    is_sold = False
                else:
                    is_sold = True
            else:
                raise ValueError(f"highest_bidder is None but num_positive_bids is {num_positive_bids}")
        else:
            if num_positive_bids == 1 and self.prev_round_max_bid < 0:
                # only one bidder in the first round
                is_sold = True
            else:
                self.prev_round_max_bid = self._highest_bid
                is_sold = num_positive_bids == 0
        return is_sold

    def _record_sale_or_pass(self) -> None:
        """Logs the outcome of the item auction and updates bidder state."""
        item = self._get_current_item()
        if self._highest_bidder_id is not None:
            winner = self._bidders[self._highest_bidder_id]
            logger.info(
                f"* Sold! {item.name} (True Value: ${item._true_value}) to {winner.name} for ${self._highest_bid}."
            )
            winner.record_win(item, self._highest_bid)
            log_entry = {
                "bidder_id": winner.id,
                "bidder_name": winner.name,
                "bid_amount": self._highest_bid,
                "true_value": item._true_value,
                "round": "Hammer Price",
            }
            self._auction_result.item_results.append(
                WonItemResult(
                    item_id=item.id,
                    true_value=item._true_value,
                    bidder_id=winner.id,
                    bid_amount=self._highest_bid,
                )
            )
        else:
            logger.info(f"* Passed! No bids received for {item.name}.")
            log_entry = {
                "bidder_id": None,
                "bidder_name": "None",
                "bid_amount": "Passed",
                "true_value": item._true_value,
                "round": "Passed",
            }
            self._auction_result.item_results.append(
                PassedItemResult(
                    item_id=item.id,
                    true_value=item._true_value,
                )
            )

        item_log_key = f"{item.name} (Start Price: ${item._original_price})"
        for round_num, bids in self._bidding_history_item.items():
            for bid in bids:
                bidder = self._bidders[bid["bidder_id"]]
                self._auction_logs[item_log_key].append(
                    {
                        "bidder_id": bidder.id,
                        "bidder_name": bidder.name,
                        "bid_amount": bid["bid_amount"] if bid["bid_amount"] >= 0 else "Withdraw",
                        "true_value": None,  # Not relevant for round bids
                        "round": round_num,
                    }
                )
        self._auction_logs[item_log_key].append(log_entry)
        self.prev_round_max_bid = -1

        for bidder_id, bidder in self._bidders.items():
            if bidder_id != self._highest_bidder_id:
                bidder.record_loss()

    def _finalize_item_auction(self) -> None:
        """Called after SUMMARIZE/REPLAN to advance internal state for all bidders."""
        for bidder in self._bidders.values():
            bidder.advance_to_next_item()

    def _get_auctioneer_bid_request_msg(self) -> str:
        """Generates the auctioneer's message to solicit bids."""
        item = self._get_current_item()
        round_num = self._current_bid_round

        if self._highest_bidder_id is None:
            if self._item_failed_to_sell_round and self._enable_discount:
                msg = (
                    f"Seeing as we've had no takers at the initial price, we're going to lower the "
                    f"starting bid to ${item.price} for {item.name} to spark some interest! "
                    f"Do I have any takers?"
                )
            else:
                # First round message
                remaining_item_names = [i.name for i in self._items[self._current_item_index :]]
                remaining_items_count = len(remaining_item_names)
                msg = (
                    f"Attention, bidders! {p.no('item', remaining_items_count)} left, they are: "
                    f"{', '.join(remaining_item_names)}.\n\n"
                    f"Now, please bid on {item.name}. "
                    f"The starting price for bidding for {item.name} is ${item.price}. Anyone interested in this item?"
                )
        else:
            # Subsequent rounds message
            last_round_history = self._get_bidding_history_text_for_round(round_num - 1)
            required_bid = item.price * self._min_markup_pct
            msg = (
                f"Thank you! This is the {p.ordinal(str(round_num + 1))} round of bidding for this item:\n"
                f"{last_round_history}\n\n"
                f"Now we have ${self._highest_bid} from Bidder {self._highest_bidder_id} for {item.name}. "
                f"The minimum increase over this highest bid is ${int(required_bid)}. "
                f"Do I have any advance on ${self._highest_bid}?"
            )

        return msg

    def _get_rebid_msg(self, bid_price: int, fail_msg: str) -> str:
        return f"Your bid of ${bid_price} failed, because {fail_msg}: You must reconsider your bid."

    def _get_hammer_msg(self) -> str:
        """Generates the auctioneer's message when an item is sold or passed."""
        item = self._get_current_item()

        if self._highest_bidder_id is None:
            return f"Since no one bid on {item.name}, we'll move on to the next item."
        else:
            winner = self._bidders[self._highest_bidder_id]
            return (
                f"Sold! {item.name} to {winner.name} at ${self._highest_bid}! "
                f"The true Value for {item.name} is ${item._true_value}."
            )

    def _get_bidding_history_text_for_round(self, round_num: int) -> str:
        """Formats bidding history for a specific round."""
        history_text = ""
        bids = self._bidding_history_item.get(round_num, [])
        if not bids:
            return "No bids placed in this round."
        for bid in bids:
            bidder_name = self._bidders[bid["bidder_id"]].name
            amount = bid["bid_amount"]
            if amount < 0:
                history_text += f"- {bidder_name} withdrew\n"
            else:
                history_text += f"- {bidder_name}: ${amount}\n"
        return history_text.strip()

    def _get_all_bidding_history_text(self) -> str:
        """Formats the complete bidding history for the current item."""
        full_history = ""
        for round_num in sorted(self._bidding_history_item.keys()):
            round_history = self._get_bidding_history_text_for_round(round_num)
            full_history += f"Round {round_num + 1}:\n{round_history}\n\n"
        return full_history.strip()

    def _get_current_item(self) -> AuctionItem:
        item = self.current_item
        if item is None:
            raise RuntimeError("Internal Error: `self.currnet_item` should be not None when calling _get_hammer_msg")
        return item

    def _is_summarize_response(self, d: dict[int, Any]) -> bool:
        return all(isinstance(v, SummarizeResponse) for v in d.values())


class AggregatedAuctionResult(pydantic.BaseModel):
    results: list[AuctionResult]
    true_skill: dict[str, tuple[float, float]]
    bidders_list: list[list[Bidder]]

    @classmethod
    def compute_ranks(cls, score_dict: dict[int, int]) -> dict[int, int]:
        # Sort players by score (high to low)
        sorted_players = sorted(score_dict.items(), key=lambda x: -x[1])

        ranks = {}
        current_rank = 0
        prev_score = None
        tie_count = 0

        for i, (player, score) in enumerate(sorted_players):
            if score == prev_score:
                tie_count += 1
            else:
                current_rank = i
                tie_count = 1
            ranks[player] = current_rank
            prev_score = score

        return ranks

    @classmethod
    def from_results(cls, results: list[AuctionResult]) -> "AggregatedAuctionResult":
        bidders_list = [result.bidders for result in results]

        bidder_id_to_idx = {bdr.id: i for i, bdr in enumerate(bidders_list[0])}
        rates = [[trueskill.Rating()] for _ in range(len(bidder_id_to_idx))]

        for result in results:
            match_result = result.calculate_result()
            score_dict = {bidder_id_to_idx[bidder_id]: items["profit"] for bidder_id, items in match_result.items()}

            ranks_dict = cls.compute_ranks(score_dict)
            ranks_list = [ranks_dict[i] for i in range(len(rates))]

            rates = trueskill.rate(rates, ranks_list)

        true_skill = dict()
        for bidder in bidders_list[0]:
            rate = rates[bidder_id_to_idx[bidder.id]][0]
            true_skill[bidder.name] = (rate.mu, rate.sigma)

        agg_result = cls(results=results, bidders_list=bidders_list, true_skill=true_skill)
        logger.info(f"Trueskill:\n{json.dumps(true_skill, indent=4)}")
        return agg_result


class AuctionTask(Task):
    def __init__(
        self,
        items: list[dict],
        bidders: list[dict[str, Any]],
        min_markup_pct: float = 0.1,
        enable_discount: bool = False,
        discount_percentage: float = 0.5,
        max_discount_rounds: int = 3,
        num_auctions: int = 1,
        item_order: ItemOrder = "random",
    ):
        self.items = items
        self.bidders = bidders
        self.min_markup_pct = min_markup_pct
        self.enable_discount = enable_discount
        self.discount_percentage = discount_percentage
        self.max_discount_rounds = max_discount_rounds

        self.num_auctions = num_auctions
        self.item_order = item_order

    async def iterate_environments(self) -> AsyncIterator[Environment[AuctionResult]]:
        for _ in range(self.num_auctions):
            yield AuctionEnvironment(
                items=copy.deepcopy(self.items),
                bidders=copy.deepcopy(self.bidders),
                min_markup_pct=self.min_markup_pct,
                enable_discount=self.enable_discount,
                discount_percentage=self.discount_percentage,
                max_discount_rounds=self.max_discount_rounds,
                item_order=self.item_order,
            )

    def aggregate_results(self, results: Sequence[AuctionResult]) -> AggregatedAuctionResult:
        return AggregatedAuctionResult.from_results(list(results))
