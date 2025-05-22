import json
from evaluator import Evaluator, ensemble_evaluate
from schemas import AgentOutputItem, Answer, BenchmarkItem


def load_llm_config(config_path:str = "llm_config.json") -> dict:
    with open(config_path, "r") as f:
        return json.load(f)

def load_agent_output_dataset(dataset_path:str = "dataset/example_agent_output.json") -> list[AgentOutputItem]:
    with open(dataset_path, "r") as f:
        agent_output_dataset = json.load(f)
        return [AgentOutputItem(**item) for item in agent_output_dataset]

async def run_evaluate(evaluator_list:list[Evaluator], agent_output_item:AgentOutputItem, to_evaluate_item:BenchmarkItem):
    answer = Answer(
        answer=agent_output_item.answer,
        reasoning_steps=agent_output_item.reasoning_list,
        function_calls=agent_output_item.tool_use_list
    )
    return await ensemble_evaluate(evaluator_list, answer, to_evaluate_item)

async def main():
    #load llm config
    llm_configs = load_llm_config()
    parse_llm_config = llm_configs["parse_llm_config"]
    evaluate_llm_configs = llm_configs["evaluate_llm_configs"]
    #load agent output dataset
    agent_output_dataset = load_agent_output_dataset()
    #load evaluate dataset
    evaluator_list: list[Evaluator] = []
    for evaluate_llm_config in evaluate_llm_configs:
        for _ in range(3):
            evaluator = Evaluator(
                dataset_path="dataset/example_evaluate_data.json",
                parse_model=parse_llm_config["model_name"],
                parse_model_api_key=parse_llm_config.get("api_key", None),
                parse_model_base_url=parse_llm_config.get("base_url", None),
                api_key=evaluate_llm_config.get("api_key", None),
                model_name=evaluate_llm_config["model_name"],
                base_url=evaluate_llm_config.get("base_url", None),
                **evaluate_llm_config.get("model_params",{})
            )
            evaluator_list.append(evaluator)
    evaluate_dataset = await evaluator.load_validate_data()
    #evaluate
    # run parallel
    for agent_output_item in agent_output_dataset:
        task_id = agent_output_item.task_id
        to_evaluate_item = [item for item in evaluate_dataset if item.task_id == task_id][0]
        answer = Answer(
            answer=agent_output_item.answer,
            reasoning_steps=agent_output_item.reasoning_list,
            function_calls=agent_output_item.tool_use_list
        )
        score,results = await ensemble_evaluate(evaluator_list, answer, to_evaluate_item)
        print(f"Task ID: {task_id}")
        print(f"Score: {score}")
        # print(results)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())