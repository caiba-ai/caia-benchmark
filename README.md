# CAIA - A benchmark for Crypto AI Agent

> Mission – Provide an open, reproducible yardstick for measuring how well AI agents reason about, interact with, and execute on crypto-native tasks and problem sets.
> 
> 
> CAIA Benchmark aims to creating domain-specific, industry-grade evaluations that move beyond generic academic sets and reflect the realities of **crypto**.
> 

## Key Takeaways
- We conducted experiments using a simple demo-agent from [Vals.ai Finance-agent](https://github.com/vals-ai/finance-agent/tree/main) equipped with common tools (Google search, web browser). We compared several foundation LLMs and evaluated their performance on entry-level crypto analyst tasks—results are available on our [Leaderboard](https://huggingface.co/spaces/cyberco/CAIA-Benchmark-Leaderboard).
- No models using the demo-agent framework achieve satisfactory performance as entry-level crypto analysts, with none scoring more than 40 out of 100.
- Most models struggled with analysis and calculation tasks due to the lack of crypto-specific tools, leading to inaccurate results (numbers, addresses, tokenomics).
- Simple agent frameworks may limit model capabilities; we encourage the use of complex and powerful agent frameworks to better challenge our benchmark.


---

## 1 Context — Why a crypto-specific benchmark?

- **Money is on the line.** Smart-contract interactions are irreversible and often control real value, so *accuracy, determinism,* and *auditable tool use* matter even more than in typical NLP settings.
- **Generic agent benchmarks miss the nuances.** Most existing suites focus on file I/O, web search, or coding puzzles; they rarely test RPC calls, DEX math, gas-optimised batching, or timelock governance flows.
- **Push the ecosystem forward.** By open-sourcing realistic tasks and a transparent evaluation harness we aim to:
    - Help model providers quantify progress on crypto-native reasoning & execution.
    - Give protocol teams a quick smoke-test for model integration.
    - Encourage research on safety, latency, and gas-aware optimization.

---

## 2 Datasets (v0.1)

| Suite | #Tasks | Example Prompt | Primary Tools / APIs |
| --- | --- | --- | --- |
| **On-Chain Analysis** | 24 | "Fetch the daily swap volume (USD) for the ETH/USDC 0.05 % pool on Uniswap V3 for 2025-01-02." | JSON-RPC, subgraph, Dune |
| **Tokenomics Deep-Dive** | 6 | "Compute OP's circulating supply, FDV, and annualised emission schedule as of block *N*." | Etherscan, DefiLlama, CSV math |
| **Project Discovery** | 8 | "Find three newly deployed restaking protocols this week and rank them by GitHub commits." | Block-explorer, GitHub API, web search |
| **Overlap** | 3 | "Give the contract address and GitHub repo for the EigenLayer AVS example mentioned in their docs." | Block-explorer, GitHub API, web search |

*Format* All tasks are in `json` with fields:

```
{
    "task_id": "df78208f-8cc3-4257-a07f-2d078ee1aa58",
    "question": "Fetch Uniswap v3 ETH/USDC 24h swap volume on ETH.",
    "level": 1,
    "evaluate": {
        "items": [
            {
                "target": "ANSWER",
                "points": 1,
                "criteria": "..."
            },
            {
                "target": "REASONING",
                "step": 1,
                "points": 2,
                "criteria": "..."
            },
            {
                "target": "REASONING",
                "step": 2,
                "points": 2,
                "criteria": "..."
            },
            {
                "target": "TOOL_USE",
                "points": 4,
                "criteria": "..."
            }
        ]
    }
}

```

---

## 3 Evaluation Methodology

We follow a **LLM-as-Judge** (and soon *LLM-as-Jury*) approach inspired by Vals AI's broader framework. 

Key points:

1. **Scoring pipeline**
    1. Collect reference answers & tool traces & reasoning steps for each task.
    2. Run candidate agent → capture *answer*, *step-level tool calls*, *arguments*, *reasoning steps*.
    3. Ask *k* different judge LLMs to grade each dimension for three times with temperature and config; take the mean.
    4. Normalise each dimension to a 0-1 scale so suites with many tool calls don't dominate.
2. **Dimensions & weights (v0.1)**
    
    
    | Dimension | Weight | Notes |
    | --- | --- | --- |
    | **Answer correctness** | 0.1-0.4 | *Exact-match* for single-truth tasks; partial credit for open-ended questions. |
    | **Reasoning validity** | 0.2-0.4 | Judges evaluate chain-of-thought (hidden to model under test). |
    | **Tool-use accuracy** | 0.2-0.4 | (a) Call matches intent (b) Params accurate (c) No unsafe ops |

3. **Aggregation**
    
    *Final score* = Σ (normalised dimension × weight).
    

> Full rubric lives in evaluator.py

Checkout our demo result on our [Leaderboard on Huggingface](https://huggingface.co/spaces/cyberco/CAIA-Benchmark-Leaderboard)

---

## 4 How to Use
1. Run the public questions with your assistant/agent sysmtem
2. Collect your assistant/agent's outputs and convert to the form we expect.
3. Run the evaluation/ Upload to our [Leaderboard on Huggingface](https://huggingface.co/spaces/cyberco/CAIA-Benchmark-Leaderboard)
4. Analyze your scores 

- ### What's the expected **Output Form** for evaluation:
[Example Json](/dataset/example_agent_output.json)

```
{
    "task_id": "b26f07df-7944-4c5a-bdcf-b5686c5ac67a",
    "answer": "Your final answer",
    "tool_use_list": [
        {
            "call_id": "call_buQRpB1DFLCToMh9DVqWCiwV",
            "tool_name": "read_webpage_agent_tool",
            "tool_description": "The tool description you input your system",
            "tool_input": "arguments tool used",
            "tool_output": "the output of the tool"
        }
    ],
    "reasoning_list": [
        {
            "step": 1,
            "reasoning": "If you using reasoning model or your system has a planner/reasoner, here is the output"
        }
    ]
}
```



### Roadmap

- v0.2 – Add more onchain execution tasks
- v0.3 – Add more dataset varieties + public leaderboard
- v1.0 – Formal spec freeze & CITATION file

### Contact

*If you have any Questions, feedback or wish to collaborate?* Open an issue or ping **@james_dai** on Telegram or @DaiZeshi on X(twitter).




