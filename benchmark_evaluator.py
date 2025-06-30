import os
import json
import asyncio
from typing import List, Optional
from .schemas import BenchmarkItem, AgentOutputItem, Answer, EvaluateScore, Usage
from .evaluator import Evaluator, ensemble_evaluate


evaluator_llm_configs =  [
        {
            "model_name": "o3-2025-04-16",
            "api_key": os.getenv("OPENAI_API_KEY", None),
            "model_params": {
                "reasoning_effort":"medium"
            }
        },
        {
            "model_name": "gpt-4.1",
            "api_key": os.getenv("OPENAI_API_KEY", None),
            "model_params": {
                "temperature": 0.2
            }
        },
        {
            "model_name": "deepseek-r1-250528",
            "api_key": os.getenv("DEEPSEEK_API_KEY", None),
            "base_url": os.getenv("DEEPSEEK_BASE_URL", None),
            "model_params": {
                "max_tokens": 16000
            }
        }
    ]

class BenchmarkEvaluator:
    def __init__(self, dataset_path: str = "internal/dataset/benchmark_data_v4.json"):
        self.dataset_path = dataset_path
        self.evaluator_list: List[Evaluator] = []
        self.benchmark_data: List[BenchmarkItem] = self.init_benchmark_data()

    def init_evaluator(self) -> List[Evaluator]:
        if self.evaluator_list:
            return self.evaluator_list
        # 初始化多个evaluator
        evaluator_list: List[Evaluator] = []
        for evaluate_llm_config in evaluator_llm_configs:
            for _ in range(3):
                evaluator = Evaluator(
                    dataset_path=self.dataset_path,
                    parse_model="gpt-4.1-mini-2025-04-14",
                    parse_model_api_key=os.getenv("OPENAI_API_KEY", None),
                    parse_model_base_url=None,
                    api_key=evaluate_llm_config.get("api_key", None),
                    model_name=evaluate_llm_config["model_name"],
                    base_url=evaluate_llm_config.get("base_url", None) if evaluate_llm_config.get("base_url", None) else None,
                    **evaluate_llm_config.get("model_params", {}),
                )
                evaluator_list.append(evaluator)
        self.evaluator_list = evaluator_list
        return evaluator_list
    
    def init_benchmark_data(self) -> List[BenchmarkItem]:
        with open(self.dataset_path, "r") as f:
            content = json.load(f)
            benchmark_items = []
            for item in content:
                try:
                    benchmark_item = BenchmarkItem(**item)
                except Exception as e:
                    print(f"Error loading benchmark item: {e}")
                    print(item)

                benchmark_items.append(benchmark_item)
            return benchmark_items
    
    def find_eval_item(self, task_id:Optional[str] = None, question:Optional[str] = None) -> BenchmarkItem | None:
        if task_id:
            return [item for item in self.benchmark_data if item.task_id == task_id] [0]
        if question:
            return [item for item in self.benchmark_data if item.question == question][0]
        return None

    async def a_evaluate(self, agent_output_item: AgentOutputItem, only_answer: bool = False) -> tuple[float, list[EvaluateScore], Usage | None]:
        self.init_evaluator()
        # 构造Answer对象
        answer = Answer(
            answer=agent_output_item.answer,
            reasoning_steps=agent_output_item.reasoning_list,
            function_calls=agent_output_item.tool_use_list,
        )
        to_evaluate_item = self.find_eval_item(
            task_id= agent_output_item.task_id if agent_output_item.task_id else None,
            question= agent_output_item.question if agent_output_item.question else None
        )
        if not to_evaluate_item:
            return 0, [], agent_output_item.usage
        # 调用ensemble_evaluate
        score, results = await ensemble_evaluate(self.evaluator_list, answer, to_evaluate_item, only_answer)
        return score, results, agent_output_item.usage
    
        # INSERT_YOUR_CODE
    async def a_batch_evaluate(self, agent_output_items: list[AgentOutputItem],  only_answer: bool = False) -> list[tuple[float, list[EvaluateScore], Usage | None]]:
        """
        批量并发评估 agent_output_items 列表
        返回每个item的(score, results)元组列表
        """
        self.init_evaluator()
        tasks = []
        for agent_output_item in agent_output_items:
            tasks.append(self.a_evaluate(agent_output_item, only_answer=only_answer))
        results = await asyncio.gather(*tasks)
        return results