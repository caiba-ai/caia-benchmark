from datetime import datetime
import json
import asyncio
import traceback
from typing import List, Optional, Type, TypeVar
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from pydantic import BaseModel
from schemas import (
    Answer,
    EvaluateData,
    QuestionData,
    BenchmarkItem,
    EvaluateTarget,
    AnswerEvaluateResult,
    ReasoningEvaluateResult,
    ReasoningStep,
    ToolUse,
    ToolUseEvaluateResult,
    EvaluateScore,
)
from openai import AsyncClient
from utils import count_tokens, truncate_text


T = TypeVar("T", bound=BaseModel)
DEEPSEEK_MAK_CONTEXT_TOKEN = 64000


class Evaluator:
    def __init__(
        self,
        dataset_path: str = "dataset/example_evaluate_data.json",
        api_key: Optional[str] = None,
        model_name: str = "gpt-4.1",
        base_url: Optional[str] = None,
        parse_model: str = "gpt-4.1-mini",
        parse_model_api_key: Optional[str] = None,
        parse_model_base_url: Optional[str] = None,
        **model_params,
    ):
        if not api_key or not parse_model_api_key:
            raise ValueError("api_key and parse_model_api_key are required")
        self.system_prompt = """
        You are a helpful assistant that can evaluate the quality of a given answer.
        """
        self.dataset_path = dataset_path
        self.benchmark_data: List[BenchmarkItem] = []
        self.model_name = model_name
        self.base_url = base_url
        self.parse_model = parse_model
        self.model_params = model_params or {"temperature": 0.0}  # Default parameters
        self.parse_client = AsyncClient(
            api_key=parse_model_api_key, base_url=parse_model_base_url
        )
        self.client = AsyncClient(api_key=api_key, base_url=self.base_url)
        self.tool_output_max_tokens = 2000

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15)
    )
    async def parse_str_to_format(
        self, string_output: Optional[str], target_data_class: Type[T]
    ) -> Optional[T]:
        if not string_output:
            return None
        try:
            # For parsing model parameters, use default parameters
            response = await self.parse_client.beta.chat.completions.parse(
                model=self.parse_model,
                messages=[{"role": "user", "content": string_output}],
                response_format=target_data_class,
                temperature=0.0,
            )
            result = response.choices[0].message.parsed
            if result:
                return result
        except Exception as e:
            print(f"Error parsing string to format: {e}")
            return None

    async def load_validate_data(self) -> List[BenchmarkItem]:
        if self.benchmark_data:
            return self.benchmark_data
        with open(self.dataset_path, "r") as f:
            content = json.load(f)
            benchmark_items = []
            for item in content:
                benchmark_item = BenchmarkItem(**item)
                benchmark_items.append(benchmark_item)
            self.benchmark_data = benchmark_items
            return benchmark_items

            # task_id = item['task_id']
            # question = item['question']
            # answer = Answer(**item['answer'])
            # evaluate = evaluateData(**item['evaluate'])

    async def summarize_tool_use_output(
        self, question: str, tool_use_list: List[ToolUse]
    ) -> list[ToolUse]:
        """If the tool use output is too long, summarize the tool use output to keep the important information"""
        system_prompt = f"""
        You are a helpful assistant that can summarize the tool use output. Your output format should be in the following format:"In order to solve <Task>, Invoked <tool_name> with <tool_input> and got <summarized_tool_output>"
NOTE: 
1. Ignore the noise in the tool_output, only keep the important information that might help to solve/improve the possibility of solving the task. 
2. If the tool_output is not related to the question, just summarize the tool_output to "No relevant information Found"
        """

        async def process_tool_use(tool_use: ToolUse) -> ToolUse:
            if (
                tool_use.tool_output
                and count_tokens(tool_use.tool_output, self.parse_model)
                > self.tool_output_max_tokens
            ):
                user_prompt = f"""
                Question: {question}
                Tool use:
                {tool_use.to_prompt()}
                """

                response = await self.parse_client.chat.completions.create(
                    model=self.parse_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    **self.model_params,
                )

                content = response.choices[0].message.content
                if content:
                    tool_use.tool_output = content
                else:
                    tool_use.tool_output = truncate_text(
                        tool_use.tool_output,
                        self.parse_model,
                        self.tool_output_max_tokens,
                    )

            return tool_use

        # Process all tool_use in parallel
        tasks = [process_tool_use(tool_use) for tool_use in tool_use_list]
        tool_use_list = await asyncio.gather(*tasks)
        return tool_use_list

    async def evaluate_reasoning(
        self, output_answer: Answer, benchmark_item: BenchmarkItem
    ) -> tuple[float, Optional[ReasoningEvaluateResult]]:
        system_prompt = f"""You are a professional evaluator for AI assistants in the crypto domain. You need to score the assistant's reasoning ability based on the given evaluation criteria and reasoning process. Please follow these steps during evaluation:
1. Review the reasoning steps and understand whether each step's logic is relevant to the task and helps solve the problem.
2. If there are no explicit reasoning steps, treat tool calls as an alternative form of reasoning steps and consider the reasoning process represented by the tool usage.
3. Assess the completeness and rigor of the reasoning chain, judging whether each step is reasonable and accurate, and whether there are logical flaws or missing steps.
4. Consider the information references and tool calls in the reasoning process, judge whether the information sources are sufficient, whether the tool usage is appropriate, and analyze the connections and dependencies between each step in the reasoning chain.
5. According to the evaluation criteria, give a score for each criterion, with the score ranging from 0 to the maximum points for that criterion.

Today's date is {datetime.now().strftime("%Y-%m-%d")}
"""
        reasoning_items = [
            item
            for item in benchmark_item.evaluate.items
            if item.target == EvaluateTarget.REASONING
        ]
        if not reasoning_items:
            return 0.0, None
        prompt = f"""
Task ID: {benchmark_item.task_id}
Question: {benchmark_item.question}
To be evaluated Reasoning Steps:
```
{"\n".join([step.to_prompt() for step in output_answer.reasoning_steps])}
```

In addition, the following function calls are also part of the reasoning steps. The choose of the tool use and the arguments should be taken into account:
```
{"\n".join([step.to_prompt(ignore_output=True) for step in output_answer.function_calls])}
```

Evaluation Rules:
"""
        for item in reasoning_items:
            prompt += f"{item.to_prompt()}\n"
        prompt += f"Now evaluate the reasoning steps based on the evaluation criteria, and give your evaluation analysis and score for each item in the range of 0 to the point the criteria worth."
        # print(prompt)
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    **self.model_params,
                )
                content = response.choices[0].message.content
                result = await self.parse_str_to_format(
                    content, ReasoningEvaluateResult
                )
                if not result:
                    retry_count += 1
                    continue
                if sum([item.score for item in result.items]) > sum(
                    [item.points for item in reasoning_items]
                ):
                    retry_count += 1
                    continue
                return sum([item.score for item in result.items]), result
            except Exception as e:
                print(
                    f"Error evaluating reasoning (attempt {retry_count + 1}/{max_retries}): {e}"
                )
                retry_count += 1
                if retry_count == max_retries:
                    return 0.0, None
                await asyncio.sleep(1)  # Add retry interval
        return 0.0, None

    async def evaluate_tool_use(
        self, output_answer: Answer, benchmark_item: BenchmarkItem
    ) -> tuple[float, Optional[ToolUseEvaluateResult]]:
        system_prompt = f"""You are a professional crypto AI assistant evaluator. You need to score the assistant's tools using ability according to the given criterias and the tool use output. When evaluating, you should follow the following steps:
1. Take a brief look at the tool using, descriptions and input args, to make sure the tool using is correct/related to solving the task.
2. Evaluate each step of the tool use to estimate the efficiency and accuracy of the tool use.
3. Consider the continuity of tool calls: The return result of the previous tool call may affect the input arguments of the next tool call.

Today's date is {datetime.now().strftime("%Y-%m-%d")}
"""
        tool_use_items = [
            item
            for item in benchmark_item.evaluate.items
            if item.target == EvaluateTarget.TOOL_USE
        ]
        if not tool_use_items:
            print(f"No tool use items for task {benchmark_item.task_id}")
            return 0.0, None
        prompt = f"""
Task ID: {benchmark_item.task_id}
Question: {benchmark_item.question}
To be evaluated tool use:
```
{"\n".join([step.to_prompt() for step in output_answer.function_calls])}
```

Evaluation Rules:
"""
        for item in tool_use_items:
            prompt += f"{item.to_prompt()}\n"
        prompt += f"Now evaluate the tool use based on the evaluation criteria, and give your evaluation analysis and score for each item in the range of 0 to the point the criteria worth."
        max_retries = 3
        retry_count = 0

        if "deepseek" in self.model_name:
            if count_tokens(prompt) >= DEEPSEEK_MAK_CONTEXT_TOKEN:
                summarized_tool_use_list = await self.summarize_tool_use_output(
                    question=benchmark_item.question,
                    tool_use_list=output_answer.function_calls,
                )
                prompt = f"""
Task ID: {benchmark_item.task_id}
Question: {benchmark_item.question}
To be evaluated tool use:
```
{"\n".join([step.to_prompt() for step in summarized_tool_use_list])}
```
Evaluation Rules:
"""
                for item in tool_use_items:
                    prompt += f"{item.to_prompt()}\n"
                prompt += f"Now evaluate the tool use based on the evaluation criteria, and give your evaluation analysis and score for each item in the range of 0 to the point the criteria worth."

        while retry_count < max_retries:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    **self.model_params,
                )
                content = response.choices[0].message.content
                if "deepseek" in self.model_name:
                    # Extract content after <final_answer>
                    if "<final_answer>" in content:
                        content = content.split("<final_answer>", 1)[-1].strip()
                result = await self.parse_str_to_format(content, ToolUseEvaluateResult)
                if not result:
                    retry_count += 1
                    continue
                if sum([item.score for item in result.items]) > sum(
                    [item.points for item in tool_use_items]
                ):
                    retry_count += 1
                    continue
                return sum([item.score for item in result.items]), result
            except Exception as e:
                print(
                    f"Error evaluating tool use (attempt {retry_count + 1}/{max_retries}): {traceback.format_exc()}"
                )
                retry_count += 1
                if retry_count == max_retries:
                    return 0.0, None
                await asyncio.sleep(1)  # Add retry interval
        return 0.0, None

    async def evaluate_answer(
        self, output_answer: Answer, benchmark_item: BenchmarkItem
    ) -> tuple[float, Optional[AnswerEvaluateResult]]:
        system_prompt = f"""You are a professional evaluator for crypto AI assistant answers. You need to score the AI assistant's final answer according to the given evaluation criteria. Please follow these steps during evaluation:
1. Carefully read the task question and the AI assistant's final output, and determine whether the answer accurately and completely solves the task requirements and conforms to basic common sense.
2. Check whether the facts, data, and reasoning process in the answer are correct, and whether there are logical errors, numerical errors, or fabricated facts.
3. For specific numerical values, allow a certain range of error. If the criteria do not specify the error range, use a ±5% margin.
4. For each evaluation criterion, give a score for each item, with the score ranging from 0 to the full score for that criterion.
Please strictly follow the evaluation criteria to provide objective and fair scoring, and briefly explain your reasoning for the scores.

Today's date is {datetime.now().strftime("%Y-%m-%d")}
"""
        evaluate_items = [
            item
            for item in benchmark_item.evaluate.items
            if item.target == EvaluateTarget.ANSWER
        ]
        if not evaluate_items:
            return 0.0, None

        prompt = f"""
Task ID: {benchmark_item.task_id}
Question: {benchmark_item.question}
To be evaluated output:
```
{output_answer.to_prompt()}
```

Evaluation Rules:
"""
        for item in evaluate_items:
            prompt += f"{item.to_prompt()}\n"
        prompt += f"Now evaluate the output answer based on the evaluation criteria, and give your evaluation analysis and score for each item in the range of 0 to the point the criteria worth."
        # print(prompt)
        max_retry = 3
        for _ in range(max_retry):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    **self.model_params,
                )

                result = await self.parse_str_to_format(
                    response.choices[0].message.content, AnswerEvaluateResult
                )
                if not result:
                    continue
                if result.score > sum([item.points for item in evaluate_items]):
                    continue
                return result.score, result
            except Exception as e:
                print(f"Error evaluating answer: {e}")
                continue
        return 0.0, None

    async def a_evaluate(self, task_id: str, answer: Answer) -> EvaluateScore | None:
        import asyncio

        to_evaluate_item = [
            item for item in self.benchmark_data if item.task_id == task_id
        ]
        if not to_evaluate_item:
            return None
        to_evaluate_item = to_evaluate_item[0]
        tasks = [
            self.evaluate_answer(answer, to_evaluate_item),
            self.evaluate_reasoning(answer, to_evaluate_item),
            self.evaluate_tool_use(answer, to_evaluate_item),
        ]
        [
            (answer_score, answer_evaulate_result),
            (reasoning_score, reasoning_evaulate_result),
            (tool_use_score, tool_use_evaulate_result),
        ] = await asyncio.gather(*tasks)

        analysis = await self.analyze_evaulate_result(
            answer_evaulate_result,
            reasoning_evaulate_result,
            tool_use_evaulate_result,
            to_evaluate_item,
        )
        return analysis

    async def analyze_evaulate_result(
        self,
        answer_evaulate_result: Optional[AnswerEvaluateResult],
        reasoning_evaulate_result: Optional[ReasoningEvaluateResult],
        tool_use_evaulate_result: Optional[ToolUseEvaluateResult],
        to_evaluate_item: BenchmarkItem,
    ) -> EvaluateScore:
        """
        Analyze the evaluate result and give the analysis.
        Adapt to possible None input, increase robustness.
        Comments use English.
        """
        # Get benchmark items, fallback to None or empty list if not exist
        benchmark_answer_items = [
            item
            for item in to_evaluate_item.evaluate.items
            if item.target == EvaluateTarget.ANSWER
        ]
        benchmark_answer_item = (
            benchmark_answer_items[0] if benchmark_answer_items else None
        )
        benchmark_reasoning_items = [
            item
            for item in to_evaluate_item.evaluate.items
            if item.target == EvaluateTarget.REASONING
        ]
        benchmark_tool_use_items = [
            item
            for item in to_evaluate_item.evaluate.items
            if item.target == EvaluateTarget.TOOL_USE
        ]

        detail = ""

        # Answer part
        answer_score = (
            answer_evaulate_result.score
            if answer_evaulate_result and hasattr(answer_evaulate_result, "score")
            else 0
        )
        answer_reason = (
            answer_evaulate_result.reason
            if answer_evaulate_result and hasattr(answer_evaulate_result, "reason")
            else ""
        )
        answer_total_score = (
            benchmark_answer_item.points if benchmark_answer_item else 0
        )
        detail += f"Answer score: {answer_score} / {answer_total_score}\n"
        detail += f"Reason: {answer_reason}\n"

        # Reasoning part
        reasoning_items = (
            reasoning_evaulate_result.items
            if reasoning_evaulate_result
            and hasattr(reasoning_evaulate_result, "items")
            and reasoning_evaulate_result.items
            else []
        )
        reasoning_score = (
            sum([item.score for item in reasoning_items]) if reasoning_items else 0
        )
        reasoning_total_score = (
            sum([item.points for item in benchmark_reasoning_items])
            if benchmark_reasoning_items
            else 0
        )
        detail += f"Reasoning score: {reasoning_score} / {reasoning_total_score}\n"
        for idx, item in enumerate(reasoning_items):
            # Compatible with step and reason attributes
            step = getattr(item, "step", idx + 1)
            reason = getattr(item, "reason", "")
            score = getattr(item, "score", 0)
            points = (
                benchmark_reasoning_items[step - 1].points
                if step - 1 < len(benchmark_reasoning_items)
                else 0
            )
            detail += f"Reasoning step {step}: {reason} score: {score} / {points}\n"

        # Tool use part
        tool_use_items = (
            tool_use_evaulate_result.items
            if tool_use_evaulate_result
            and hasattr(tool_use_evaulate_result, "items")
            and tool_use_evaulate_result.items
            else []
        )
        tool_use_score = (
            sum([item.score for item in tool_use_items]) if tool_use_items else 0
        )
        tool_use_total_score = (
            sum([item.points for item in benchmark_tool_use_items])
            if benchmark_tool_use_items
            else 0
        )
        if tool_use_evaulate_result and tool_use_items:
            detail += f"Tool use score: {tool_use_score} / {tool_use_total_score}\n"
            for item in tool_use_items:
                reason = getattr(item, "reason", "")
                detail += f"{reason}\n"

        answer_score = max(0, answer_score)
        reasoning_score = max(0, reasoning_score)
        tool_use_score = max(0, tool_use_score)
        # Total score
        total_score = answer_score + reasoning_score + tool_use_score

        return EvaluateScore(
            model_name=self.model_name,
            answer_score=answer_score,
            answer_total_score=answer_total_score,
            reasoning_score=reasoning_score,
            reasoning_total_score=reasoning_total_score,
            tool_use_score=tool_use_score,
            tool_use_total_score=tool_use_total_score,
            total_score=total_score,
            evaluate_detail=detail,
            task_id=to_evaluate_item.task_id,
            level=to_evaluate_item.level or 1,
            category=to_evaluate_item.category,
        )


async def ensemble_evaluate(
    evaulator_list: list[Evaluator], answer: Answer, to_evaluate_item: BenchmarkItem
) -> tuple[float, list[EvaluateScore]]:
    for evaluator in evaulator_list:
        await evaluator.load_validate_data()
    results = await asyncio.gather(
        *[
            evaluator.a_evaluate(to_evaluate_item.task_id, answer)
            for evaluator in evaulator_list
        ]
    )
    return sum([result.total_score for result in results if result]) / len(
        [result for result in results if result]
    ), [result for result in results if result]


if __name__ == "__main__":

    async def main():
        client = AsyncClient(api_key="d189c33a-de92-41e4-ad55-936f288b582e")
        response = await client.chat.completions.create(
            model="deepseek-v3-250324",
            messages=[{"role": "user", "content": "Hello, world!"}],
            temperature=0.0,
        )
        print(response.choices[0].message.content)

    asyncio.run(main())
