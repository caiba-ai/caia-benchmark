import argparse
import asyncio
import json
import os
from datetime import datetime

from demo.agent import agent_logger
from demo.get_agent import get_agent
from demo.tools import tool_logger
from tqdm.asyncio import tqdm
from pydantic import BaseModel


class InputQuestion(BaseModel):
    question:str
    task_id:str

from schemas import QuestionData



async def run_tests_parallel(
    output_dir,
    questions:list[QuestionData]=[],
    model_name="anthropic/claude-3-7-sonnet-20250219",
    max_concurrent=5,
    save_results=False,
    parameters={},
):
    """Run multiple questions in parallel using the custom model"""
    agent = await get_agent(model_name, parameters)

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_question(question):
        async with semaphore:
            return await agent.run(question)

    tasks = [process_question(question.question) for question in questions]

    results = await tqdm.gather(*tasks, desc="Processing questions")

    formatted_results = []
    for i, (question, result) in enumerate(zip(questions, results)):
        if isinstance(result, Exception):
            formatted_results.append(
                {"question": question.question, "task_id": question.task_id, "level": question.level, "category": question.category, "success": False, "error": str(result)}
            )
        else:
            formatted_results.append(
                {"question": question.question, "task_id": question.task_id, "level": question.level, "category": question.category, "success": True, "result": result}
            )

    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"results_test_{timestamp}.json")

        with open(output_file, "w") as f:
            json.dump(formatted_results, f, indent=2)

    return formatted_results



if __name__ == "__main__":
    import asyncio
    import json
    import os

    # 加载问题数据
    with open("dataset/benchmark_tasks.json", "r") as f:
        questions_data = json.load(f)
    
    questions = [QuestionData(**item) for item in questions_data]
    # 提取所有问题
    # result = asyncio.run(run_tests_parallel(
    #     output_dir="results",
    #     questions=questions,
    #     model_name="volcengine/deepseek-r1-250120",
    #     max_concurrent=5,
    #     save_results=True,
    # ))
    result = asyncio.run(run_tests_parallel(
        output_dir="results",
        questions=questions,
        model_name="openrouter/anthropic/claude-opus-4",
        max_concurrent=5,
        save_results=True,
    ))

