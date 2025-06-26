
import json
from typing import List, Optional
from schemas import BenchmarkItem, AgentOutputItem, Answer, EvaluateScore
from evaluator import Evaluator, ensemble_evaluate

class BenchmarkEvaluator:
    def __init__(self, llm_config_path: str = "llm_config.json", dataset_path: str = "internal/dataset/benchmark_data_v4.json"):
        self.llm_config_path = llm_config_path
        self.dataset_path = dataset_path
        self.evaluator_list: List[Evaluator] = []
        self.benchmark_data: List[BenchmarkItem] = self.init_benchmark_data()

    def init_evaluator(self) -> List[Evaluator]:
        if self.evaluator_list:
            return self.evaluator_list
        # 加载llm配置
        with open(self.llm_config_path, "r", encoding="utf-8") as f:
            llm_configs = json.load(f)
        parse_llm_config = llm_configs["parse_llm_config"]
        evaluate_llm_configs = llm_configs["evaluate_llm_configs"]
        # 初始化多个evaluator
        evaluator_list: List[Evaluator] = []
        for evaluate_llm_config in evaluate_llm_configs:
            for _ in range(3):
                evaluator = Evaluator(
                    dataset_path=self.dataset_path,
                    parse_model=parse_llm_config["model_name"],
                    parse_model_api_key=parse_llm_config.get("api_key", None),
                    parse_model_base_url=parse_llm_config.get("base_url", None),
                    api_key=evaluate_llm_config.get("api_key", None),
                    model_name=evaluate_llm_config["model_name"],
                    base_url=evaluate_llm_config.get("base_url", None),
                    **evaluate_llm_config.get("model_params", {}),
                )
                evaluator_list.append(evaluator)
        self.evaluator_list = evaluator_list
        return evaluator_list
    
    def init_benchmark_data(self) -> List[BenchmarkItem]:
        if self.benchmark_data:
            return self.benchmark_data
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
            self.benchmark_data = benchmark_items
            return benchmark_items
    
    def find_eval_item(self, task_id:Optional[str] = None, question:Optional[str] = None) -> BenchmarkItem | None:
        if task_id:
            return [item for item in self.benchmark_data if item.task_id == task_id] [0]
        if question:
            return [item for item in self.benchmark_data if item.question == question][0]
        return None

    async def a_evaluate(self, agent_output_item: AgentOutputItem) -> tuple[float, list[EvaluateScore]]:
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
            return 0, []
        # 调用ensemble_evaluate
        score, results = await ensemble_evaluate(self.evaluator_list, answer, to_evaluate_item)
        return score, results