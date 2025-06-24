
import json
from typing import List
from schemas import BenchmarkItem, AgentOutputItem, Answer
from evaluator import Evaluator, ensemble_evaluate

class BenchmarkEvaluator:
    def __init__(self, llm_config_path: str = "llm_config.json", dataset_path: str = "internal/dataset/benchmark_data_v4.json"):
        self.llm_config_path = llm_config_path
        self.dataset_path = dataset_path
        self.evaluator_list: List[Evaluator] = []
        self.benchmark_data: List[BenchmarkItem] = []

    def init_evaluator(self):
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

    async def evaluate(self, agent_output_item: AgentOutputItem, to_evaluate_item: BenchmarkItem):
        # 构造Answer对象
        answer = Answer(
            answer=agent_output_item.answer,
            reasoning_steps=agent_output_item.reasoning_list,
            function_calls=agent_output_item.tool_use_list,
        )
        # 调用ensemble_evaluate
        score, results = await ensemble_evaluate(self.evaluator_list, answer, to_evaluate_item)
        return score, results