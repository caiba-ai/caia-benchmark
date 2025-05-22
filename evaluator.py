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
    Answer, EvaluateData, QuestionData, BenchmarkItem, 
    EvaluateTarget, AnswerEvaluateResult, ReasoningEvaluateResult, ReasoningStep, ToolUse, ToolUseEvaluateResult,
    EvaluateScore
)
from openai import AsyncClient
from utils import count_tokens, truncate_text



T = TypeVar("T", bound=BaseModel)


class Evaluator:
    def __init__(self, 
                 dataset_path:str = 'dataset/example_evaluate_data.json',
                 api_key:Optional[str] = None,
                 model_name:str = "gpt-4.1",
                 base_url:Optional[str] = None,
                 parse_model:str = "gpt-4.1-mini",
                 parse_model_api_key:Optional[str] = None,
                 parse_model_base_url:Optional[str] = None,
                 **model_params):
        if not api_key or not parse_model_api_key:
            raise ValueError("api_key and parse_model_api_key are required")
        self.system_prompt = """
        You are a helpful assistant that can evaluate the quality of a given answer.
        """
        self.dataset_path = dataset_path
        self.benchmark_data:List[BenchmarkItem] = []
        self.model_name = model_name
        self.base_url = base_url
        self.parse_model = parse_model
        self.model_params = model_params or {"temperature": 0.0}  # 默认参数
        self.parse_client = AsyncClient(api_key=parse_model_api_key, base_url=parse_model_base_url)
        self.client = AsyncClient(api_key=api_key, base_url=self.base_url)
        self.tool_output_max_tokens = 2000

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    async def parse_str_to_format(self, string_output:Optional[str], target_data_class:  Type[T]) -> Optional[T]:
        if not string_output:
            return None
        try:
            # 对于解析模型的参数，使用默认参数
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
    
    async def summarize_tool_use_output(self, question:str, tool_use_list:List[ToolUse]) -> list[ToolUse]:
        """If the tool use output is too long, summarize the tool use output to keep the important information"""
        system_prompt = f"""
        You are a helpful assistant that can summarize the tool use output. Your output format should be in the following format:"In order to solve <Task>, Invoked <tool_name> with <tool_input> and got <summarized_tool_output>"
NOTE: 
1. Ignore the noise in the tool_output, only keep the important information that might help to solve/improve the possibility of solving the task. 
2. If the tool_output is not related to the question, just summarize the tool_output to "No relevant information Found"
        """
        async def process_tool_use(tool_use: ToolUse) -> ToolUse:
            if count_tokens(tool_use.tool_output, self.parse_model) > self.tool_output_max_tokens:
                user_prompt = f"""
                Question: {question}
                Tool use:
                {tool_use.to_prompt()}
                """
                
                response = await self.parse_client.chat.completions.create(
                    model=self.parse_model,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    **self.model_params
                )
                
                content = response.choices[0].message.content
                if content:
                    tool_use.tool_output = content
                else:
                    tool_use.tool_output = truncate_text(tool_use.tool_output, self.parse_model, self.tool_output_max_tokens)
            
            return tool_use

        # 并行处理所有tool_use
        tasks = [process_tool_use(tool_use) for tool_use in tool_use_list]
        tool_use_list = await asyncio.gather(*tasks)
        return tool_use_list


    async def evaluate_reasoning(self, output_answer:Answer, benchmark_item:BenchmarkItem) -> tuple[float, Optional[ReasoningEvaluateResult]]:
        reasoning_items = [item for item in benchmark_item.evaluate.items if item.target == EvaluateTarget.REASONING]
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
        prompt += f"Now evaluate the reasoning steps based on the evaluation criteria, and give the score for each item in the range of 0 to the point the criteria worth."
        # print(prompt)
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **self.model_params
                )
                content = response.choices[0].message.content
                result = await self.parse_str_to_format(content, ReasoningEvaluateResult)
                if not result:
                    retry_count += 1
                    continue
                if sum([item.score for item in result.items]) > sum([item.points for item in reasoning_items]):
                    retry_count += 1
                    continue
                return sum([item.score for item in result.items]), result
            except Exception as e:
                print(f"Error evaluating reasoning (attempt {retry_count + 1}/{max_retries}): {e}")
                retry_count += 1
                if retry_count == max_retries:
                    return 0.0, None
                await asyncio.sleep(1)  # 添加重试间隔
        return 0.0, None

    async def evaluate_tool_use(self, output_answer:Answer, benchmark_item:BenchmarkItem) -> tuple[float, Optional[ToolUseEvaluateResult]]:
        tool_use_items = [item for item in benchmark_item.evaluate.items if item.target == EvaluateTarget.TOOL_USE]
        if not tool_use_items:
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
        prompt += f"Now evaluate the tool use based on the evaluation criteria, and give the score for each item in the range of 0 to the point the criteria worth."
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **self.model_params
                )
                content = response.choices[0].message.content
                result = await self.parse_str_to_format(content, ToolUseEvaluateResult)
                if not result:
                    retry_count += 1
                    continue
                if sum([item.score for item in result.items]) > sum([item.points for item in tool_use_items]):
                    retry_count += 1
                    continue
                return sum([item.score for item in result.items]), result
            except Exception as e:
                print(f"Error evaluating tool use (attempt {retry_count + 1}/{max_retries}): {traceback.format_exc()}")
                retry_count += 1
                if retry_count == max_retries:
                    return 0.0, None
                await asyncio.sleep(1)  # 添加重试间隔
        return 0.0, None

    
    async def evaluate_answer(self, output_answer:Answer, benchmark_item:BenchmarkItem) -> tuple[float, Optional[AnswerEvaluateResult]]:
        evaluate_items = [item for item in benchmark_item.evaluate.items if item.target == EvaluateTarget.ANSWER]
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
        prompt += f"Now evaluate the output answer based on the evaluation criteria, and give the score for each item in the range of 0 to the point the criteria worth."
        # print(prompt)
        max_retry = 3
        for _ in range(max_retry):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **self.model_params
                )


                result = await self.parse_str_to_format(response.choices[0].message.content, AnswerEvaluateResult)
                if not result:
                    continue
                if result.score > sum([item.points for item in evaluate_items]):
                    continue
                return result.score, result
            except Exception as e:
                print(f"Error evaluating answer: {e}")
                continue
        return 0.0, None

    async def a_evaluate(self, task_id:str, answer:Answer) -> EvaluateScore | None:
        import asyncio
        to_evaluate_item = [item for item in self.benchmark_data if item.task_id == task_id]
        if not to_evaluate_item:
            return None
        to_evaluate_item = to_evaluate_item[0]
        tasks = [
            self.evaluate_answer(answer, to_evaluate_item),
            self.evaluate_reasoning(answer, to_evaluate_item),
            self.evaluate_tool_use(answer, to_evaluate_item),
        ]
        [(answer_score, answer_evaulate_result), (reasoning_score, reasoning_evaulate_result), (tool_use_score, tool_use_evaulate_result)] = await asyncio.gather(*tasks)
        
        analysis = await self.analyze_evaulate_result(answer_evaulate_result, reasoning_evaulate_result, tool_use_evaulate_result, to_evaluate_item)
        return analysis
    

    async def analyze_evaulate_result(self, 
                                      answer_evaulate_result:AnswerEvaluateResult, 
                                      reasoning_evaulate_result:ReasoningEvaluateResult, 
                                      tool_use_evaulate_result:ToolUseEvaluateResult, 
                                      to_evaluate_item:BenchmarkItem) -> EvaluateScore:
        """Analyze the evaulate result and give the analysis"""
        benchmark_answer_item = [item for item in to_evaluate_item.evaluate.items if item.target == EvaluateTarget.ANSWER][0]
        benchmark_reasoning_items = [item for item in to_evaluate_item.evaluate.items if item.target == EvaluateTarget.REASONING]
        benchmark_tool_use_items = [item for item in to_evaluate_item.evaluate.items if item.target == EvaluateTarget.TOOL_USE]
        detail = ""
        detail += f"Answer score: {answer_evaulate_result.score} / {benchmark_answer_item.points}\n"
        detail += f"Reason: {answer_evaulate_result.reason}\n"
        detail += f"Reasoning score: {sum([item.score for item in reasoning_evaulate_result.items])} / {sum([item.points for item in benchmark_reasoning_items])}\n"
        for item in reasoning_evaulate_result.items:
            detail += f"Reasoning step {item.step}: {item.reason} score: {item.score} / {benchmark_reasoning_items[item.step-1].points}\n"
        detail += f"Tool use score: {sum([item.score for item in tool_use_evaulate_result.items])} / {sum([item.points for item in benchmark_tool_use_items])}\n"
        for item in tool_use_evaulate_result.items:
            detail += f"{item.reason}\n"
        return EvaluateScore(
            model_name=self.model_name,
            answer_score=answer_evaulate_result.score,
            answer_total_score=benchmark_answer_item.points,
            reasoning_score=sum([item.score for item in reasoning_evaulate_result.items]),
            reasoning_total_score=sum([item.points for item in benchmark_reasoning_items]),
            tool_use_score=sum([item.score for item in tool_use_evaulate_result.items]),
            tool_use_total_score=sum([item.points for item in benchmark_tool_use_items]),
            total_score=answer_evaulate_result.score + sum([item.score for item in reasoning_evaulate_result.items]) + sum([item.score for item in tool_use_evaulate_result.items]),
            evaluate_detail=detail
        )
        

async def ensemble_evaluate(evaulator_list:list[Evaluator], answer:Answer, to_evaluate_item:BenchmarkItem) -> tuple[float, list[EvaluateScore]]:
    for evaluator in evaulator_list:
        await evaluator.load_validate_data()
    results = await asyncio.gather(*[evaluator.a_evaluate(to_evaluate_item.task_id, answer) for evaluator in evaulator_list])
    return sum([result.total_score for result in results if result]) / len([result for result in results if result]), [result for result in results if result]


# if __name__ == "__main__":
#     evaluator = Evaluator()
#     benchmark_data = asyncio.run(evaluator.load_validate_data())
#     fake_answer = Answer(
#         answer="42 Celsius degrees in Paris today", 
#         reasoning_steps=[
#             ReasoningStep(
#                 step=1, 
#                 reasoning="Paris is the capital of France, so I need to find the temperature in Paris", 
#             ),
#             ReasoningStep(
#                 step=2, 
#                 reasoning="I need to use web search to find the temperature in Paris",
#             )
#         ],
#         function_calls=[
#             ToolUse(
#                 call_id="1",
#                 tool_name="web_search",
#                 tool_description="Use the google to do web search",
#                 tool_input="temperature in Paris",
#                 tool_output="42 Celsius degrees"
#             )
#         ]
#     )
#     benchmark_data = asyncio.run(ensemble_evaluate(fake_answer, evaluator.benchmark_data[0]))
#     print(benchmark_data)


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