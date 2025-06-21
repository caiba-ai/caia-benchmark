import asyncio
import json
from statistics import mean

from evaluator import Evaluator, ensemble_evaluate
from schemas import AgentOutputItem, Answer, BenchmarkItem, EvaluateScore

input = "eval_results_0621_17:09:41"


def load_llm_config(config_path: str = "llm_config.json") -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def load_agent_output_dataset(
    dataset_path: str = "dataset/example_agent_output.json",
) -> list[AgentOutputItem]:
    with open(dataset_path, "r") as f:
        agent_output_dataset = json.load(f)
        return [AgentOutputItem(**item) for item in agent_output_dataset]


async def run_evaluate(
    evaluator_list: list[Evaluator],
    agent_output_item: AgentOutputItem,
    to_evaluate_item: BenchmarkItem,
):
    answer = Answer(
        answer=agent_output_item.answer,
        reasoning_steps=agent_output_item.reasoning_list,
        function_calls=agent_output_item.tool_use_list,
    )
    return await ensemble_evaluate(evaluator_list, answer, to_evaluate_item)


async def main():
    # Add concurrency control parameters
    MAX_CONCURRENT_TASKS = 10  # Can be adjusted based on API limits
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    # load llm config
    llm_configs = load_llm_config()
    parse_llm_config = llm_configs["parse_llm_config"]
    evaluate_llm_configs = llm_configs["evaluate_llm_configs"]
    # load agent output dataset
    agent_output_dataset = load_agent_output_dataset(
        f"inputs/eval_results/{input}.json"
    )
    # load evaluate dataset
    evaluator_list: list[Evaluator] = []
    for evaluate_llm_config in evaluate_llm_configs:
        for _ in range(3):
            evaluator = Evaluator(
                dataset_path="internal/dataset/benchmark_data_v4.json",
                parse_model=parse_llm_config["model_name"],
                parse_model_api_key=parse_llm_config.get("api_key", None),
                parse_model_base_url=parse_llm_config.get("base_url", None),
                api_key=evaluate_llm_config.get("api_key", None),
                model_name=evaluate_llm_config["model_name"],
                base_url=evaluate_llm_config.get("base_url", None),
                **evaluate_llm_config.get("model_params", {}),
            )
            evaluator_list.append(evaluator)
    evaluate_dataset = await evaluator.load_validate_data()

    # Use semaphore to control concurrent evaluation function
    async def run_evaluate_task_with_semaphore(
        agent_output_item: AgentOutputItem,
        to_evaluate_item: BenchmarkItem,
        task_index: int,
    ) -> tuple[str, str, float, list[EvaluateScore]]:
        async with semaphore:
            print(
                f"Starting to process task {task_index + 1}/{len(agent_output_dataset)}: {agent_output_item.task_id}"
            )
            try:
                task_id = agent_output_item.task_id
                answer = Answer(
                    answer=agent_output_item.answer,
                    reasoning_steps=agent_output_item.reasoning_list,
                    function_calls=agent_output_item.tool_use_list,
                )
                score, results = await ensemble_evaluate(
                    evaluator_list, answer, to_evaluate_item
                )
                print(
                    f"Completed task {task_index + 1}/{len(agent_output_dataset)}: {task_id}, score: {score}"
                )
                return task_id, to_evaluate_item.question, score, results
            except Exception as e:
                print(f"Error processing task {task_index + 1}: {e}")
                # Return default values to avoid program crash
                return agent_output_item.task_id, to_evaluate_item.question, 0.0, []

    # Create all tasks
    tasks = []
    for idx, agent_output_item in enumerate(agent_output_dataset):
        task_id = agent_output_item.task_id
        to_evaluate_item = [
            item for item in evaluate_dataset if item.task_id == task_id
        ][0]
        tasks.append(
            run_evaluate_task_with_semaphore(agent_output_item, to_evaluate_item, idx)
        )

    # Execute all tasks in parallel, but controlled by semaphore
    print(
        f"Starting evaluation of {len(tasks)} tasks, maximum concurrency: {MAX_CONCURRENT_TASKS}"
    )
    results: list[tuple[str, str, float, list[EvaluateScore]]] = await asyncio.gather(
        *tasks
    )

    # Process results
    print(f"Evaluation completed, successfully processed {len(results)} tasks")

    for index, (task_id, question, score, result_list) in enumerate(results):
        print(f"Task ID: {task_id}")
        print(f"Question: {question}")
        print(f"Score: {score}")

        if result_list:  # Ensure result_list is not empty
            answer_score = mean(
                [
                    item.answer_score / item.answer_total_score
                    for item in result_list
                    if item.answer_total_score > 0
                ]
            )
            reasoning_score = mean(
                [
                    item.reasoning_score / item.reasoning_total_score
                    for item in result_list
                    if item.reasoning_total_score > 0
                ]
            )
            tool_use_score = mean(
                [
                    item.tool_use_score / item.tool_use_total_score
                    for item in result_list
                    if item.tool_use_total_score > 0
                ]
            )
        else:
            answer_score = reasoning_score = tool_use_score = 0.0

        import csv
        import os

        # Define CSV filename
        csv_filename = f"results/{input}.csv"
        # Check if file does not exist to determine if header should be written
        file_exists = os.path.isfile(csv_filename)

        # Write header only if file does not exist
        with open(csv_filename, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(
                    [
                        "task_id",
                        "question",
                        "score",
                        "answer_score",
                        "reasoning_score",
                        "tool_use_score",
                        "results",
                        "agent_output",
                    ]
                )
            # Serialize results to string to avoid formatting issues
            import json

            results_str = json.dumps(
                [r.__dict__ if hasattr(r, "__dict__") else str(r) for r in result_list],
                ensure_ascii=False,
            )
            agent_output_str = agent_output_dataset[index].to_prompt()

            # Clean newlines to avoid automatic line breaks in CSV
            agent_output_str = agent_output_str.replace("\n", " ").replace("\r", " ")
            question = question.replace("\n", " ").replace(
                "\r", " "
            )  # Also process question field
            results_str = results_str.replace("\n", " ").replace(
                "\r", " "
            )  # Also process results field

            writer.writerow(
                [
                    task_id,
                    question,
                    round(score * 10, 2),
                    round(answer_score * 100, 2),
                    round(reasoning_score * 100, 2),
                    round(tool_use_score * 100, 2),
                    results_str,
                    agent_output_dataset[index].answer,
                ]
            )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
