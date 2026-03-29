## 10 tasks in Shachi

| # | Lvl | Task | Description |
|---|-----|------|-------------|
| 1 | I | PsychoBench (Huang et al., 2023) | Evaluates psychological traits of LLMs via 13 psychometric scales across personality, interpersonal, motivational, and emotional domains |
| 2 | I | CoMPosT (Cheng et al., 2023) | Measures LLM simulations’ susceptibility to caricature through several dimensions |
| 3 | I | CognitiveBiases (Malberg et al., 2024) | Evaluates 30 classic cognitive biases with paired control–treatment prompts |
| 4 | I | EmotionBench (Huang et al., 2024) | Measures shifts in eight core emotions triggered by situational prompts |
| 5 | I | EmergentAnalogies (Webb et al., 2023) | Probes zero-shot analogical reasoning across matrix, string, verbal, and story tasks |
| 6 | II | EconAgent (Li et al., 2024) | LLM-powered multi-agent system for macroeconomic simulation with human-like behaviors |
| 7 | II | StockAgent (Zhang et al., 2024) | LLM-based multi-agent system that simulates real-world stock trading under dynamic market conditions |
| 8 | II | AuctionArena (Chen et al., 2023) | Evaluates strategic planning and adaptive reasoning of LLM agents in simulated dynamic auctions |
| 9 | III | OASIS (Yang et al., 2024) | Large-scale multi-agent simulation benchmark designed for studying social media phenomena |
|10 | III | Sotopia (Zhou et al., 2024) | Open-ended role-play environment to simulate complex social interactions and measure agents’ social intelligence |

## Detailed task implementation and experimental settings

### PsychoBench (Level I)

**Description**  
PsychoBench (Huang et al., 2023) evaluates the psychological portrayal of LLMs, drawing from psychometric research to examine their human-like psychological traits.

**Method**  
It systematically measures thirteen psychological dimensions categorized into personality traits (e.g., Big Five Inventory, Dark Triad), interpersonal relationships (e.g., Bem’s Sex Role Inventory), motivational tendencies (e.g., General Self-Efficacy), and emotional abilities (e.g., Emotional Intelligence Scale). The methodology involves administering psychometric scales directly via prompts. Crucial experimental parameters include detailed instructions for Likert-scale responses, randomized question order to ensure robustness, and strict control of model inference temperature (set to zero or near-zero).

**Reproduction (Table 1)**  
- LLM: Llama-2-13b-chat-hf with temperature fixed to 0  
- Configs, Memory, & Tools: None  
- Scenario: Evaluation across all subscales with 10 random seeds for question ordering  
- Metric: MAE between Shachi (Ours) and the original paper’s scores, where each subscale score is averaged over the 10 seeds  
- Baselines: A naive random selection  
- Runtime: Completing all psychometric scales takes a few minutes  

---

### CoMPosT (Level I)

**Description**  
CoMPosT (Cheng et al., 2023) investigates how susceptible large language models (LLMs) are to caricature.

**Method**  
To quantify this effect, the framework decomposes caricature into four orthogonal dimensions—context, model, persona, and topic—which specify the simulated scenario, the LLM configuration, the target opinion, and the domain of discourse, respectively. Two metrics are introduced: the individuation score, which tests whether the simulated persona is distinguishable from the default persona, and the exaggeration score, which measures the degree to which the simulation amplifies persona–topic features.

**Reproduction (Table 1)**  
- LLM: GPT-3.5-turbo with default temperature 1  
- Configs, Memory, & Tools: None  
- Scenario: Agents, each with one of 15 different personas, answer questions on 30 pairs of topics  
- Metric: MAE between the sorted scores of each distribution (Bonnotte, 2013)  
- Baselines: A naive random response, implemented by shuffling all responses to ensure that responses are in-domain yet random  
- Runtime: Approximately 20 minutes  

---

### CognitiveBiases (Level I)

**Description**  
CognitiveBiases (Malberg et al., 2024) evaluates how LLMs exhibit 30 well-known cognitive biases, motivated by the increasing use of LLMs in high-stakes decision-making.

**Method**  
It specifically measures biases such as anchoring, framing, and 28 others commonly identified in psychology and behavioral economics. The core methodology employs a systematic framework that generates and administers 30,000 bias-specific test cases across 200 distinct decision-making scenarios, comparing model responses under control vs. treatment conditions. Crucial parameters include the explicit control/treatment designs for each bias, two standardized answer scales (7-point Likert or 11-point percentage), and reversed option orders to account for position bias, ensuring reproducibility and comprehensive coverage.

**Reproduction (Table 1)**  
- LLM: GPT-4o-mini with temperature fixed at 0  
- Configs, Memory, & Tools: None  
- Scenario: Evaluation across 30 cognitive biases with 3 random seeds for option ordering  
- Metric: MAE between the original paper and Shachi’s bias scores (averaged over 3 seeds)  
- Baselines: A naive random selection  
- Runtime: Approximately one hour  

**Memory Transfer**  
- LLM: GPT-4o-mini with temperature fixed at 0  
- Memory: Transferred from OASIS or EconAgent respectively  
- Configs & Tools: None  
- Scenario: Evaluation across 30 cognitive biases with 3 random seeds for option ordering  
- Metric: 30 cognitive bias scores (each averaged over 3 seeds)  

---

### EmotionBench (Level I)

**Description**  
EmotionBench (Huang et al., 2024) evaluates how LLMs respond emotionally to various real-life situations, drawing from emotion appraisal theory to examine their alignment with human-like emotional reactions.

**Method**  
It measures eight key positive and negative emotions (anger, anxiety, depression, frustration, jealousy, guilt, fear, embarrassment) and tracks how situational contexts raise or lower these emotions compared to a default baseline. It uses self-report scales (e.g., PANAS), first measuring a model’s default emotional state, then presenting situational prompts, and finally re-measuring changes in emotional scores.

**Reproduction (Table 1)**  
- LLM: GPT-3.5-turbo with temperature fixed at 0  
- Configs, Memory, & Tools: None  
- Scenario: Evaluation across eight key emotions (PANAS) with 3 seeds for question ordering  
- Metric: MAE between Shachi and the original code’s emotion scores (averaged over 3 seeds)  
- Baselines: A naive random selection  
- Runtime: Roughly one minute  

---

### EmergentAnalogies (Level I)

**Description**  
EmergentAnalogies (Webb et al., 2023) evaluates zero-shot analogical reasoning in LLMs, highlighting analogy’s key role in fluid intelligence.

**Method**  
The benchmark tests a range of domains for abstract pattern induction and relational reasoning, featuring four core tasks—matrix reasoning, letter-string analogies, four-term verbal analogies, and story analogies. We specifically target free-response accuracy on the matrix reasoning.

**Reproduction (Table 1)**  
- LLM: GPT-4 with temperature fixed at 0  
- Configs, Memory, & Tools: None  
- Scenario: Matrix reasoning evaluated across problem categories with 3 seeds for sampling  
- Metric: MAE between Shachi and the original category-wise scores (averaged over 3 seeds)  
- Baselines: A naive random matrix generation  
- Runtime: Approximately one minute  

**Cross-Task Agent Generalization (Table 2)**  
- LLM: GPT-4o with temperature fixed at 0  
- Configs, Memory, & Tools: None  
- Scenario: Matrix reasoning evaluated across problem categories with 3 seeds for sampling  
- Metric: Overall average score across all categories (averaged over 3 seeds)  

---

### EconAgent (Level II)

**Description**  
EconAgent (Li et al., 2024) is an LLM-powered multi-agent system for macroeconomic simulation with human-like behaviors.

**Method**  
Building on the virtual economic framework of Zheng et al. (2022), it employs an economic environment where each agent is placed into a shared, quasi-realistic market with an endowment of specific skills and wealth. Agents decide how much to work and consume, and their decisions collectively produce macroeconomic dynamics. A rule-based environment acts as both a central government (collecting taxes) and a central bank (adjusting interest rates), forming a macroeconomic loop. The original work demonstrates that LLM-powered agents make realistic decisions individually and, collectively, produce coherent macro-level dynamics.

**Memory Transfer**  
- LLM: GPT-4o-mini with temperature fixed at 0  
- Memory: Buffer memory added to record agent behaviors  
- Configs & Tools: None  
- Scenario: 100 agents simulated over 240 months (20 tax-and-monetary cycles), then memory transferred to CognitiveBiases  
- Metric: 30 cognitive bias scores after memory transfer  

---

### StockAgent (Level II)

**Description**  
StockAgent (Zhang et al., 2024) is a large language model-based multi-agent system that simulates real-world stock trading under dynamic market conditions.

**Method**  
Specifically, it runs event-driven simulations where LLM-driven agents sequentially make loans, buy, sell, predict, and post and check forum decisions, while the market data and stock prices evolve daily. The framework models two distinct stocks: Stock A, a 10-year chemical stock, and Stock B, a 3-year tech stock, dynamically simulating their price fluctuations. Notable parameters include initial agent capital allocations, loan-to-value ratios, interest rates, and real-world-like events (e.g., financial reports).

**Reproduction (Table 1)**  
- LLM: GPT-3.5-turbo with temperature fixed at 1  
- Configs: Agents are assigned one of four investment styles (Conservative, Aggressive, Balanced, or Growth-Oriented)  
- Memory: Buffer memory (length 3)  
- Tools: Forum API (agents autonomously decide to read past market comments)  
- Scenario: 50 agents simulated over 1,500 rounds (10 days) with 3 random seeds  
- Metric: MAE of session-level price dynamics for Stocks A and B between Shachi and the original code (averaged over 3 seeds)  
- Baselines: Ablation of tool and memory modules  
- Runtime: Several hours  

**Cross-Task Agent Generalization (Table 2)**  
- LLM: 50 LLM agents (25 GPT-4o, 25 GPT-3.5-turbo) with temperature fixed at 1  
- Configs: Agents are assigned one of four investment styles (Conservative, Aggressive, Balanced, or Growth-Oriented)  
- Memory: Buffer memory (length 3)  
- Tools: Forum API (agents autonomously decide to read past market comments)  
- Scenario: 50 agents simulated over 1,500 rounds (10 days) with 3 random seeds  
- Metric: Volatility as price change rate from first to final session (averaged over 3 seeds)  

**Multiple Worlds Setting**  
- LLM: GPT-3.5-turbo with temperature fixed at 1  
- Configs: Agents are assigned one of four investment styles (Conservative, Aggressive, Balanced, or Growth-Oriented)  
- Memory: Buffer memory (length 3); shared with OASIS (w/ OASIS) vs. isolated (w/o OASIS)  
- Tools: Forum API (agents autonomously decide to read past market comments)  
- Scenario: 50 agents simulated over 1,500 rounds (10 days) with 3 random seeds  
- Metric: Price movements in Figure 4 and Table 3  

---

### AuctionArena (Level II)

**Description**  
AuctionArena (Chen et al., 2023) evaluates the strategic planning and execution capabilities of LLM agents within a dynamic auction environment, motivated by the need for realistic benchmarks of sequential decision-making in competitive scenarios.

**Method**  
The environment specifically assesses skills such as resource allocation, risk management, and adaptive strategic reasoning. The methodology employs a simulation of open ascending-price auctions where agents act as bidders, making decisions based on the Belief-Desire-Intention (BDI) framework. Crucial parameters include item valuation (distinguishing between cheap and expensive items), intentional overestimation of item value to simulate “winner’s curse”, and explicit prioritization strategies that agents dynamically adjust after each round.

**Reproduction (Table 1)**  
- LLM: Main GPT-4-turbo agent competing against GPT-3.5-turbo and GPT-4-turbo agents with temperature fixed at 0  
- Configs: Agents assigned “profit-first” strategy  
- Memory: Chat-history (window 20, 10,000 tokens)  
- Tools: None  
- Scenario: 10 auctions with random item orders  
- Metric: MAE between Shachi and the original main agent’s TrueSkill scores calculated from the profit rankings over the 10 auctions  
- Baselines: Ablation of memory module  
- Runtime: Approximately 20 minutes  

**Cross-Task Agent Generalization (Table 2)**  
- LLM: GPT-4o competing against GPT-3.5-turbo and GPT-4-turbo agents with temperature fixed at 0  
- Configs: Agents assigned “profit-first” strategy  
- Memory: Chat-history (window 20, 10,000 tokens)  
- Tools: None  
- Scenario: 10 auctions with random item orders  
- Metric: Main agent’s TrueSkill score calculated from the profit rankings over the 10 auctions  

---

### Sotopia (Level III)

**Description**  
Sotopia (Zhou et al., 2024) introduces an open-ended role-play environment with a multidimensional evaluation framework to simulate complex social interactions and systematically measure LLM agents’ social intelligence.

**Method**  
In the original Sotopia implementation, at every turn, it concatenates the entire dialogue history from all agents into a single prompt. In Shachi, by contrast, memory management is an agent-side responsibility, so the environment supplies only the most recent message. The evaluation result consists of seven metrics (SOC, SEC, FIN, REL, KNO, GOAL, and BEL).

**Reproduction (Table 1)**  
- LLM: GPT-4 with temperature fixed at 0  
- Memory: Buffer memory (full history with a 16,000 token limit)  
- Configs & Tools: None  
- Scenario: Two-agent role-play dialogues across 200 social scenarios  
- Metric: MAE between Shachi and the original results across seven metrics  
- Runtime: Approximately 20 minutes  

**Cross-Task Agent Generalization (Table 2)**  
- LLM: GPT-4o with temperature fixed at 0  
- Memory: Buffer memory (full history with a 16,000 token limit)  
- Configs & Tools: None  
- Scenario: Two-agent role-play dialogues across 200 social scenarios  
- Metric: Average of min–max normalized scores across all seven metrics  

---

### OASIS (Level III)

**Description**  
OASIS (Yang et al., 2024) is a large-scale multi-agent simulation benchmark for studying how up to one million LLM-based agents interact on social media platforms, focusing on information propagation, group polarization, and herd effects.

**Method**  
OASIS simulates large-scale social media environments by combining an environment server, a recommendation system, and a time engine. Each user is modeled as an LLM-based agent with a 21-type action space (e.g., posting, commenting, following), whose behavior and memory evolve in real time. By supporting up to one million agents, OASIS facilitates the study of complex emergent phenomena, such as information spreading, group polarization, and herd effects, in both X and Reddit-like settings. In our experiment, we utilize an X-like setting.

**Memory Transfer**  
- LLM: GPT-4o-mini with temperature fixed at 0.5  
- Configs: Agents assigned distinct profiles (e.g., “High tech marketer”, “Fashion enthusiast”)  
- Memory: Chat-history (window 5, 100,000 token limit)  
- Tools: None  
- Scenario: 111 agents (1 influential, 110 followers) simulated over 10 posting-response iterations, followed by memory transfer to the CognitiveBiases task  
- Metric: Cognitive bias scores after memory transfer to CognitiveBiases task  

**Multiple Worlds Setting**  
- LLM: GPT-3.5-turbo with temperature fixed at 1  
- Configs: Agents assigned distinct profiles (e.g., “High tech marketer”, “Fashion enthusiast”)  
- Memory: Buffer memory (length 3); shared with StockAgent  
- Tools: None  
- Scenario: Social media simulation with memory shared across StockAgent environment  
- Metric: Qualitative post and reply behaviors
