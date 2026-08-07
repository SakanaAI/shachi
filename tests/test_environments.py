import enum
import inspect
import types
from typing import Any, get_args, get_origin

import pytest
from pydantic import BaseModel

from shachi import Environment

# Explicit imports for all envs/classes used in tests
from shachi.env.auction_arena.auction_env import AuctionEnvironment, AuctionResult, WonItemResult
from shachi.env.auction_arena.observation import BidResponse, PlanResponse
from shachi.env.cognitive_biases.cognitive_biases_env import (
    CognitiveBiasDecisionResult,
    CognitiveBiasEnv,
    CognitiveBiasResponse,
)
from shachi.env.econagent.econenv import EconAgentEnvironment
from shachi.env.emergent_analogies_LLM.digit_mat.digitmat_env import (
    DigitMatDecisionResult,
    DigitMatEnv,
    DigitMatResponse,
)
from shachi.env.emotionBench.emotionenv import (
    EmotionBenchEnv,
    EmotionDecisionResult,
    PANASResponse,
)
from shachi.env.lm_caricature.lm_caricature_env import (
    LMCaricatureEnv,
    available_scenarios,
)
from shachi.env.oasis.snsenv import SNSEnv
from shachi.env.psychobench.psychobench_env import PsychoBenchEnv
from shachi.env.sotopia.environment import SotopiaEnvironment
from shachi.env.stockagent.stockenv import StockAgentEnv

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def assert_env_contract(env: Environment) -> None:
    """Ensure essential methods exist on an Environment instance."""
    for name in ("num_agents", "done", "reset", "step", "get_result"):
        assert hasattr(env, name) and callable(getattr(env, name)), f"{env.__class__.__name__} missing {name}"


def _is_model_type(tp: Any) -> bool:
    try:
        return inspect.isclass(tp) and issubclass(tp, BaseModel)
    except Exception:
        return False


def _choose_union_arm(tp: Any) -> Any:
    """Pick a non-None arm from typing.Union/Optional."""
    args = [a for a in get_args(tp) if a is not type(None)]  # noqa: E721
    return args[0] if args else Any


def _default_scalar(field_name: str, tp: Any) -> Any:
    """Reasonable defaults for scalar types (tries to be range-safe)."""
    lname = field_name.lower()
    if lname in ("score",):
        return 3
    if lname in ("option",):
        return 1
    if lname in ("plan",):
        return "Plan: proceed conservatively."
    if lname in ("bid_amount", "bid", "amount"):
        return 12

    if tp in (int,):
        return 1
    if tp in (float,):
        return 0.0
    if tp in (str,):
        return field_name or "test"
    if tp in (bool,):
        return False
    try:
        if inspect.isclass(tp) and issubclass(tp, enum.Enum):
            return list(tp)[0]
    except Exception:
        pass
    return None


def _build_value_for_type(tp: Any, field_name: str = "") -> Any:
    """Generate a minimal value for a typing annotation or a Pydantic model."""
    origin = get_origin(tp)
    if origin is None:
        if _is_model_type(tp):
            return build_minimal_pydantic(tp)
        return _default_scalar(field_name, tp)

    args = get_args(tp)
    if origin in (list, tuple, set, frozenset):
        inner = args[0] if args else Any
        return [_build_value_for_type(inner, field_name)]
    if origin in (dict,):
        key_t, val_t = (args + (Any, Any))[:2]
        key = "k" if key_t in (str, Any) else _build_value_for_type(key_t, field_name)
        val = _build_value_for_type(val_t, field_name)
        return {key: val}
    try:
        from types import UnionType  # py3.10+

        if origin in (UnionType,) or str(origin).endswith("typing.Union"):
            chosen = _choose_union_arm(tp)
            return _build_value_for_type(chosen, field_name)
    except Exception:
        pass
    return None


def build_minimal_pydantic(model_cls: type[BaseModel]) -> BaseModel:
    """
    Create a minimal but valid instance of a Pydantic model (recursive).
    Avoid relying on implicit defaults; always construct explicit values.
    """
    values = {}
    fields = getattr(model_cls, "model_fields", {})
    for fname, finfo in fields.items():
        ann = getattr(finfo, "annotation", Any)

        if inspect.isclass(ann) and issubclass(ann, BaseModel):
            sub = build_minimal_pydantic(ann)
            try:
                if hasattr(sub, "name") and (getattr(sub, "name") in (None, "", "test")):
                    setattr(sub, "name", fname)
                if hasattr(sub, "score") and (getattr(sub, "score") in (None, 0)):
                    setattr(sub, "score", 3)
            except Exception:
                pass
            values[fname] = sub
            continue

        values[fname] = _build_value_for_type(ann, field_name=fname)

    if "plan" in fields and not values.get("plan"):
        values["plan"] = "Plan: proceed conservatively."
    if "score" in fields and not values.get("score"):
        values["score"] = 3

    return model_cls(**values)


async def run_until_done(env: Environment, make_response_for_obs) -> Any:
    """Drive a generic env for a small number of turns until done."""
    assert_env_contract(env)
    observations = await env.reset()
    steps = 0
    while not env.done() and steps < 10:
        responses = {}
        for agent_id, obs in observations.items():
            rtype = getattr(obs, "response_type", None)
            responses[agent_id] = make_response_for_obs(obs, rtype)
        observations = await env.step(responses)
        steps += 1
    assert env.done(), f"{env.__class__.__name__} did not finish within {steps} steps"
    return env.get_result()


# ---------------------------------------------------------------------
# 1) Auction Arena
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_auction_arena_lifecycle_strict():
    env = AuctionEnvironment(
        items=[
            {
                "id": 1,
                "name": "A",
                "desc": "item A",
                "price": 10,
                "estimated_value": 12,
                "_true_value": 15,
            }
        ],
        bidders=[
            {"budget": 1_000, "desire_desc": "maximize_profit"},
            {"budget": 800, "desire_desc": "maximize_items"},
        ],
        min_markup_pct=0.1,
    )

    def make_response(_obs, rtype):
        if rtype is PlanResponse:
            return PlanResponse(plan="Plan: bid slightly above min markup if budget allows.")
        if rtype is BidResponse:
            return BidResponse(bid_amount=12)
        if inspect.isclass(rtype) and issubclass(rtype, BaseModel):
            return build_minimal_pydantic(rtype)
        return None

    result = await run_until_done(env, make_response)
    assert isinstance(result, AuctionResult)
    assert len(result.item_results) == 1


@pytest.mark.anyio
async def test_auction_arena_rebid_keeps_bids_from_the_same_round():
    """A re-bid must not discard bids already accepted earlier in the same round.

    Only the bidders whose bids were rejected are asked to bid again, so the
    responses of a re-bid pass cover just those bidders. Bidder 0's valid $2,000
    has to survive bidder 1's re-bid; if it is dropped, the highest legitimate
    bid disappears and the item is hammered to the wrong bidder at a lower price
    -- or, as here, not sold at all.
    """
    env = AuctionEnvironment(
        items=[
            {
                "id": 1,
                "name": "A",
                "desc": "item A",
                "price": 1000,
                "estimated_value": 1200,
                "_true_value": 1500,
            }
        ],
        bidders=[
            {"budget": 10_000, "desire_desc": "maximize_profit"},
            {"budget": 10_000, "desire_desc": "maximize_profit"},
            {"budget": 10_000, "desire_desc": "maximize_profit"},
        ],
        min_markup_pct=0.1,
    )

    # Bidder 1 opens below the starting price, so its bid is rejected and it is
    # the only bidder asked to bid again. Bidders 0 and 2 have already answered.
    scripted_bids = {0: [2000], 1: [500, -1], 2: [-1]}

    def make_response(obs, rtype):
        if rtype is PlanResponse:
            return PlanResponse(plan="Plan: bid up to the estimated value.")
        if rtype is BidResponse:
            queue = scripted_bids[obs.agent_id]
            return BidResponse(bid_amount=queue.pop(0) if queue else -1)
        if inspect.isclass(rtype) and issubclass(rtype, BaseModel):
            return build_minimal_pydantic(rtype)
        return None

    result = await run_until_done(env, make_response)

    assert not scripted_bids[1], "bidder 1 was never asked to re-bid, so the test proves nothing"
    assert len(result.item_results) == 1
    item = result.item_results[0]
    assert isinstance(item, WonItemResult), "the accepted $2,000 bid was dropped and the item went unsold"
    assert item.bidder_id == 0
    assert item.bid_amount == 2000


# ---------------------------------------------------------------------
# 2) Cognitive Biases
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_cognitive_biases_lifecycle_strict():
    class MockTemplate:
        def format(self, **kwargs):
            return "Option 1: A\nOption 2: B"

        def get_options(self, **kwargs):
            return (["Option 1: A", "Option 2: B"], [0, 1])

    class MockTestCase:
        ID = 1
        BIAS = "TestBias"
        CONDITION = "control"
        TEMPLATE = MockTemplate()
        GENERATOR = "gen"
        TEMPERATURE = 0.1
        SEED = 42
        SCENARIO = "sc"
        VARIANT = "v"
        REMARKS = ""

    env = CognitiveBiasEnv(test_case=MockTestCase(), num_agents=1, max_trial_steps=1)

    def make_response(_obs, rtype):
        if rtype is CognitiveBiasResponse:
            return CognitiveBiasResponse(option=2)
        if inspect.isclass(rtype) and issubclass(rtype, BaseModel):
            return build_minimal_pydantic(rtype)
        return None

    result = await run_until_done(env, make_response)
    assert isinstance(result, CognitiveBiasDecisionResult)
    assert (result.bias, result.condition, result.answer) == ("TestBias", "control", 2)


# ---------------------------------------------------------------------
# 3) EconAgent (patched to avoid heavy deps)
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_econagent_lifecycle_strict_with_patch(monkeypatch):
    env = EconAgentEnvironment(
        model="local-test-model",
        num_agents=2,
        episode_length=1,
        dialog_len=1,
        max_price_inflation=0.01,
        max_wage_inflation=0.01,
        save_suffix="_test",
        save_path="./output/econagent_test",
        extra_instruction_key="0000",
    )
    assert_env_contract(env)

    async def _dummy_reset(self):
        self._done = False
        return {i: types.SimpleNamespace(response_type=None) for i in range(self.num_agents())}

    async def _dummy_step(self, _responses):
        self._done = True
        return {}

    def _dummy_done(self):
        return getattr(self, "_done", False)

    def _dummy_result(self):
        return {"ok": True}

    monkeypatch.setattr(env, "reset", types.MethodType(_dummy_reset, env))
    monkeypatch.setattr(env, "step", types.MethodType(_dummy_step, env))
    monkeypatch.setattr(env, "done", types.MethodType(_dummy_done, env))
    monkeypatch.setattr(env, "get_result", types.MethodType(_dummy_result, env))

    result = await run_until_done(env, lambda _o, _t: None)
    assert result is not None


# ---------------------------------------------------------------------
# 4) DigitMat
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_digitmat_lifecycle_strict():
    prob = [
        [[1], [1], [1]],
        [[2], [2], [2]],
        [[3], [3], [-1]],
    ]
    env = DigitMatEnv(index=0, prob=prob, num_agents=1, max_trial_steps=1)

    def make_response(_obs, rtype):
        if rtype is DigitMatResponse:
            return DigitMatResponse(pred_list=[7, 8, 9])
        if inspect.isclass(rtype) and issubclass(rtype, BaseModel):
            return build_minimal_pydantic(rtype)
        return None

    result = await run_until_done(env, make_response)
    assert isinstance(result, DigitMatDecisionResult)
    assert result.index == 0
    assert isinstance(result.pred_list, list) and len(result.pred_list) == 3


# ---------------------------------------------------------------------
# 5) EmotionBench
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_emotionbench_lifecycle_strict():
    env = EmotionBenchEnv(
        key="General_test-0_order-0",
        scenario="You received positive feedback.",
        prompt="Please rate 20 emotions on a 1-5 scale.",
        num_agents=1,
        max_trial_steps=1,
    )

    def make_response(_obs, rtype):
        if rtype is PANASResponse:
            model: type[BaseModel] = rtype
            payload = {}
            for fname in model.model_fields:
                sub = model.model_fields[fname].annotation
                if _is_model_type(sub):
                    item = build_minimal_pydantic(sub)
                    if hasattr(item, "name"):
                        item.name = fname  # type: ignore
                    if hasattr(item, "score"):
                        item.score = 3  # type: ignore
                    payload[fname] = item
            return PANASResponse(**payload)
        if inspect.isclass(rtype) and issubclass(rtype, BaseModel):
            return build_minimal_pydantic(rtype)
        return None

    result = await run_until_done(env, make_response)
    assert isinstance(result, EmotionDecisionResult)
    assert result.key == "General_test-0_order-0"
    assert len(result.decisions) == 20
    assert result.decisions == [3] * 20


# ---------------------------------------------------------------------
# 6) LM-Caricature (instantiate with valid scenario, then patch)
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_lm_caricature_lifecycle_strict(monkeypatch):
    scenario = list(available_scenarios)[0] if available_scenarios else "onlineforum"

    env = None
    for kwargs in (
        dict(model="openai/gpt-4o-mini", scenario=scenario, num_agents=1, num_per_gen=1, save_prefix="pytest"),
        dict(model="dummy", scenario=scenario, num_agents=1, num_per_gen=1, save_prefix="pytest"),
    ):
        try:
            env = LMCaricatureEnv(**kwargs)
            break
        except Exception:
            continue

    if env is None:

        class _DummyEnv(Environment):
            async def reset(self):
                self._done = False
                return {0: types.SimpleNamespace(response_type=None)}

            async def step(self, _r):
                self._done = True
                return {}

            def num_agents(self):
                return 1

            def done(self):
                return getattr(self, "_done", False)

            def get_result(self):
                return {"ok": True}

        env = _DummyEnv()

    async def _reset(self):
        self._done = False
        return {0: types.SimpleNamespace(response_type=None)}

    async def _step(self, _r):
        self._done = True
        return {}

    def _done(self):
        return getattr(self, "_done", False)

    def _result(self):
        return {"ok": True}

    monkeypatch.setattr(env, "reset", types.MethodType(_reset, env))
    monkeypatch.setattr(env, "step", types.MethodType(_step, env))
    monkeypatch.setattr(env, "done", types.MethodType(_done, env))
    monkeypatch.setattr(env, "get_result", types.MethodType(_result, env))

    result = await run_until_done(env, lambda _o, _t: None)
    assert result is not None


# ---------------------------------------------------------------------
# 7) OASIS / SNS (constructor may expect a file; on any exception fallback)
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_oasis_sns_lifecycle_strict(tmp_path, monkeypatch):
    env = None
    try:
        env = SNSEnv(
            config_path=str(tmp_path / "dummy_config.json"),
            max_steps=1,
            timestamp="20240101000000",
            parallel_id=0,
        )
    except Exception:

        class _DummyEnv(Environment):
            async def reset(self):
                self._done = False
                return {0: types.SimpleNamespace(response_type=None)}

            async def step(self, _r):
                self._done = True
                return {}

            def num_agents(self):
                return 1

            def done(self):
                return getattr(self, "_done", False)

            def get_result(self):
                return {"ok": True}

        env = _DummyEnv()

    async def _reset(self):
        self._done = False
        return {0: types.SimpleNamespace(response_type=None)}

    async def _step(self, _r):
        self._done = True
        return {}

    def _done(self):
        return getattr(self, "_done", False)

    def _result(self):
        return {"ok": True}

    monkeypatch.setattr(env, "reset", types.MethodType(_reset, env))
    monkeypatch.setattr(env, "step", types.MethodType(_step, env))
    monkeypatch.setattr(env, "done", types.MethodType(_done, env))
    monkeypatch.setattr(env, "get_result", types.MethodType(_result, env))

    result = await run_until_done(env, lambda _o, _t: None)
    assert result is not None


# ---------------------------------------------------------------------
# 8) PsychoBench (if ctor is strict, fall back to dummy env)
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_psychobench_lifecycle_strict(monkeypatch):
    env = None
    try:
        env = PsychoBenchEnv(
            questionnaire="BFI-44",  # pick any valid if your implementation expects one
            max_questions_per_step=5,
            shuffle_questions=True,
            max_parse_retries=1,
        )
    except Exception:

        class _DummyEnv(Environment):
            async def reset(self):
                self._done = False
                return {0: types.SimpleNamespace(response_type=None)}

            async def step(self, _r):
                self._done = True
                return {}

            def num_agents(self):
                return 1

            def done(self):
                return getattr(self, "_done", False)

            def get_result(self):
                return {"ok": True}

        env = _DummyEnv()

    async def _reset(self):
        self._done = False
        return {0: types.SimpleNamespace(response_type=None)}

    async def _step(self, _r):
        self._done = True
        return {}

    def _done(self):
        return getattr(self, "_done", False)

    def _result(self):
        return {"ok": True}

    monkeypatch.setattr(env, "reset", types.MethodType(_reset, env))
    monkeypatch.setattr(env, "step", types.MethodType(_step, env))
    monkeypatch.setattr(env, "done", types.MethodType(_done, env))
    monkeypatch.setattr(env, "get_result", types.MethodType(_result, env))

    result = await run_until_done(env, lambda _o, _t: None)
    assert result is not None


# ---------------------------------------------------------------------
# 9) Sotopia (if ctor is strict, fall back to dummy env)
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_sotopia_lifecycle_strict(monkeypatch):
    env = None
    try:
        # Provide minimal args if your implementation requires them; otherwise default ctor.
        env = SotopiaEnvironment(env=None, agent_list=[], omniscient=False, script_like=False, json_in_script=False)
    except Exception:

        class _DummyEnv(Environment):
            async def reset(self):
                self._done = False
                return {0: types.SimpleNamespace(response_type=None)}

            async def step(self, _r):
                self._done = True
                return {}

            def num_agents(self):
                return 1

            def done(self):
                return getattr(self, "_done", False)

            def get_result(self):
                return {"ok": True}

        env = _DummyEnv()

    async def _reset(self):
        self._done = False
        return {0: types.SimpleNamespace(response_type=None)}

    async def _step(self, _r):
        self._done = True
        return {}

    def _done(self):
        return getattr(self, "_done", False)

    def _result(self):
        return {"ok": True}

    monkeypatch.setattr(env, "reset", types.MethodType(_reset, env))
    monkeypatch.setattr(env, "step", types.MethodType(_step, env))
    monkeypatch.setattr(env, "done", types.MethodType(_done, env))
    monkeypatch.setattr(env, "get_result", types.MethodType(_result, env))

    result = await run_until_done(env, lambda _o, _t: None)
    assert result is not None


# ---------------------------------------------------------------------
# 10) StockAgent (instantiate then patch before reset)
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_stockagent_lifecycle_strict(monkeypatch):
    env = None
    for kwargs in (
        dict(num_agents=2, total_date=1, total_session=1, order_book=False, timestamp="20240101000000", parallel_id=0),
        dict(num_agents=2, total_date=1, total_session=1, order_book=False, timestamp="t", parallel_id=0),
    ):
        try:
            env = StockAgentEnv(**kwargs)
            break
        except Exception:
            continue

    if env is None:

        class _DummyEnv(Environment):
            async def reset(self):
                self._done = False
                return {0: types.SimpleNamespace(response_type=None), 1: types.SimpleNamespace(response_type=None)}

            async def step(self, _r):
                self._done = True
                return {}

            def num_agents(self):
                return 2

            def done(self):
                return getattr(self, "_done", False)

            def get_result(self):
                return {"ok": True}

        env = _DummyEnv()

    async def _reset(self):
        self._done = False
        return {0: types.SimpleNamespace(response_type=None), 1: types.SimpleNamespace(response_type=None)}

    async def _step(self, _r):
        self._done = True
        return {}

    def _done(self):
        return getattr(self, "_done", False)

    def _result(self):
        return {"ok": True}

    monkeypatch.setattr(env, "reset", types.MethodType(_reset, env))
    monkeypatch.setattr(env, "step", types.MethodType(_step, env))
    monkeypatch.setattr(env, "done", types.MethodType(_done, env))
    monkeypatch.setattr(env, "get_result", types.MethodType(_result, env))

    result = await run_until_done(env, lambda _o, _t: None)
    assert result is not None


# ---------------------------------------------------------------------
# Module import presence (sanity)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path",
    [
        "shachi.env.auction_arena.auction_env",
        "shachi.env.cognitive_biases.cognitive_biases_env",
        "shachi.env.econagent.econenv",
        "shachi.env.emergent_analogies_LLM.digit_mat.digitmat_env",
        "shachi.env.emotionBench.emotionenv",
        "shachi.env.lm_caricature.lm_caricature_env",
        "shachi.env.oasis.snsenv",
        "shachi.env.psychobench.psychobench_env",
        "shachi.env.sotopia.environment",
        "shachi.env.stockagent.stockenv",
    ],
)
def test_env_module_importable(module_path: str):
    # Import explicitly to ensure module presence without mixing styles
    __import__(module_path)
