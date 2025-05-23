from math import isclose
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum

class EvaluateTarget(Enum):
    ANSWER = "ANSWER"
    REASONING = "REASONING"
    TOOL_USE = "TOOL_USE"
    SOURCES = "SOURCES"

class ToolUse(BaseModel):
    call_id: str
    tool_name:str
    tool_description: str
    tool_input:str
    tool_output: str

    def to_prompt(self, ignore_output:bool = False) -> str:
        prompt = f"Tool Name: {self.tool_name}\n"
        prompt += f"Tool Description: {self.tool_description}\n"
        prompt += f"Tool Input: {self.tool_input}\n"
        if not ignore_output:
            prompt += f"Tool Output: {self.tool_output}\n"
        return prompt


class ReasoningStep(BaseModel):
    step: int
    reasoning: Optional[str] = None
    # function_call: Optional[ToolUse] = None

    def to_prompt(self) -> str:
        prompt = f"Step {self.step}:\n"
        if self.reasoning:
            prompt += f"Reasoning: {self.reasoning}\n"
        # if self.function_call:
        #     prompt += f"Function Call: {self.function_call.to_prompt()}\n"
        return prompt

class Answer(BaseModel):
    answer: str
    reasoning_steps: List[ReasoningStep]
    function_calls: List[ToolUse]
    # sources: List[str]

    def to_prompt(self) -> str:
        prompt = f"Final Answer: {self.answer}\n"
        return prompt



class EvaluateItem(BaseModel):
    step: Optional[int] = None
    target: EvaluateTarget
    points: float
    criteria: str

    def to_prompt(self) -> str:
        prompt = f"Step {self.step}:\n" if self.step else ""
        prompt += f"Worth Points: {self.points}\n"
        prompt += f"Criteria content: {self.criteria}\n"
        return prompt

class EvaluateData(BaseModel):
    items: List[EvaluateItem]
    
    @field_validator('items')
    @classmethod
    def validate_total_points(cls, items: List[EvaluateItem]) -> List[EvaluateItem]:
        total_points = sum(item.points for item in items)
        if abs(total_points - 10.0) != 0:
            raise ValueError(f"所有评估项的权重总和必须等于10，当前总和为: {total_points}")
        return items


class QuestionData(BaseModel):
    task_id: str
    question: str
    level: int
    category: str
    # tools:Optional[List[str]] = Field(description="The tools that can be used to answer the question")

    def to_prompt(self) -> str:
        prompt = f"Task ID: {self.task_id}\n"
        prompt += f"Question: {self.question}\n"
        return prompt



class BenchmarkItem(BaseModel):
    task_id: str
    question: str = Field(description="The question to be answered")
    # answer: Answer = Field(description="The agent system output")
    evaluate: EvaluateData = Field(description="The evaluation result")



class AnswerEvaluateResult(BaseModel):
    reason: Optional[str] = None
    score: float = Field(description="The score of the answer worth")

    def __str__(self) -> str:
        return f"Reason: {self.reason}\nScore: {self.score}"
    

class ReasoningEvaluateItem(BaseModel):
    step: int
    reason: Optional[str] = None
    score: float = Field(description="The score of the reasoning step worth")

    def __str__(self) -> str:
        return f"Step: {self.step}\nReason: {self.reason}\nScore: {self.score}"

class ReasoningEvaluateResult(BaseModel):
    items: List[ReasoningEvaluateItem]

    def __str__(self) -> str:
        return "\n".join([item.__str__() for item in self.items])


class ToolUseEvaluateItem(BaseModel):
    reason: Optional[str] = None
    score: float = Field(description="The score of the tool use worth")

    def __str__(self) -> str:
        return f"Reason: {self.reason}\nScore: {self.score}"

class ToolUseEvaluateResult(BaseModel):
    items: List[ToolUseEvaluateItem]

    def __str__(self) -> str:
        return "\n".join([item.__str__() for item in self.items])



class AgentOutputItem(BaseModel):
    task_id: str
    answer: str
    tool_use_list: List[ToolUse]
    reasoning_list: List[ReasoningStep]

    def to_prompt(self) -> str:
        prompt = f"Task ID: {self.task_id}\n"
        prompt += f"Answer: {self.answer}\n"
        prompt += f"Tool Use List: {self.tool_use_list}\n"
        prompt += f"Reasoning List: {self.reasoning_list}\n"
        return prompt

class EvaluateScore(BaseModel):
    answer_total_score: float = Field(description="The total score of the answer worth")
    reasoning_total_score: float = Field(description="The total score of the reasoning worth")
    tool_use_total_score: float = Field(description="The total score of the tool use worth")

    answer_score: float = Field(description="The score of the agent get from the answer")
    reasoning_score: float = Field(description="The score of the agent get from the reasoning")
    tool_use_score: float = Field(description="The score of the agent get from the tool use")

    total_score: float = Field(description="The total score of the agent")

    evaluate_detail:Optional[str] = Field(description="The detail of the evaluation")
    model_name: str


    # @field_validator('total_score')
    @field_validator('answer_score', 'reasoning_score', 'tool_use_score')
    def non_negative(cls, v):
        if v < 0:
            raise ValueError('score cannot be negative')
        return v

    @field_validator('answer_score')
    def check_answer_score(cls, v, info):
        max_score = info.data.get('answer_total_score', 0)
        if v > max_score:
            raise ValueError('answer_score cannot exceed answer_total_score')
        return v

    @field_validator('reasoning_score')
    def check_reasoning_score(cls, v, info):
        max_score = info.data.get('reasoning_total_score', 0)
        if v > max_score:
            raise ValueError('reasoning_score cannot exceed reasoning_total_score')
        return v

    @field_validator('tool_use_score')
    def check_tool_use_score(cls, v, info):
        max_score = info.data.get('tool_use_total_score', 0)
        if v > max_score:
            raise ValueError('tool_use_score cannot exceed tool_use_total_score')
        return v
    
    @model_validator(mode='after')
    def check_totals(self):
        # 可选：限制总分（如果业务就是固定 10 分）
        if self.total_score > 10:
            raise ValueError('total_score cannot exceed 10')

        expected = self.answer_score + self.reasoning_score + self.tool_use_score
        if not isclose(self.total_score, expected, abs_tol=1e-6):
            raise ValueError(
                f'total_score ({self.total_score}) must equal the sum of '
                f'answer_score + reasoning_score + tool_use_score ({expected})'
            )
        return self