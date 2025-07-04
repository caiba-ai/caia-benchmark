import asyncio
import json
import os
from abc import ABC, abstractmethod
import traceback
from typing import Any

import backoff
from anthropic import AsyncAnthropic
from logger import get_logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from demo.utils import is_token_limit_error
from anthropic.types import Message
import httpx
provider_args = {
    "openai": {"api_key": os.getenv("OPENAI_API_KEY")},
    "google": {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "anthropic": {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "base_url": "https://api.anthropic.com/v1/",
    },
    "together": {
        "api_key": os.getenv("TOGETHER_API_KEY"),
        "base_url": "https://api.together.xyz/v1/",
    },
    "fireworks": {
        "api_key": os.getenv("FIREWORKS_API_KEY"),
        "base_url": "https://api.fireworks.ai/v1/",
    },
    "mistralai": {
        "api_key": os.getenv("MISTRAL_API_KEY"),
        "base_url": "https://api.mistral.ai/v1/",
    },
    "grok": {
        "api_key": os.getenv("GROK_API_KEY"),
        "base_url": "https://api.x.ai/v1",
    },
    "cohere": {
        "api_key": os.getenv("COHERE_API_KEY"),
        "base_url": "https://api.cohere.ai/compatibility/v1/",
    },
    "volcengine": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://ark.cn-beijing.volces.com/api/v3/",
    },
    "openrouter":{
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1/",
    },
    "fireworks": {
        "api_key": os.getenv("FIREWORKS_API_KEY"),
        "base_url": "https://api.fireworks.ai/inference/v1/",
    },
}

llm_logger = get_logger(__name__)


class LLM(ABC):
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        params = self.get_provider_args()
       
        self.client = AsyncOpenAI(
            api_key=params["api_key"], base_url=params.get("base_url")
        )
        # self.provider = provider if provider == "anthropic" else "openai"

    def get_provider_args(self) -> dict[str, Any]:
        params = provider_args[self.provider]
        return params

    def chat(self, conversation: list[dict[str, Any]]) -> str:
        pass

    @abstractmethod
    def parse_response(self, response: dict[str, Any]) -> str:
        pass

    @abstractmethod
    def append_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_content: Any,
        tool_result: str,
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def get_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        pass

    def convert_usage(self, usage: dict[str, Any]) -> dict[str, Any]:
        pass


class GeneralLLM(LLM):
    def __init__(
        self,
        provider: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        super().__init__(provider=provider, model_name=model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        if provider == "anthropic":
            params = self.get_provider_args()
            proxy_url = "http://127.0.0.1:7890"
    
            http_client = httpx.AsyncClient(proxy=proxy_url)
            self.anthropic_client = AsyncAnthropic(api_key=params["api_key"], http_client=http_client)

    async def safe_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] = [],
        ignore_token_error: bool = False,
    ) -> ChatCompletion:
        while True:
            try:
                # Use the retryable chat function with original messages
                return await self._retryable_chat(messages, tools)
            except Exception as e:
                error_str = str(e).lower()

                # Handle token limit errors specially
                if is_token_limit_error(error_str):
                    if ignore_token_error:
                        raise Exception(f"Token limit error: {str(e)}")
                    if len(messages) > 2:
                        llm_logger.warning(
                            f"Too long, removing oldest message pair. {len(messages)}"
                        )
                        messages.pop(1)
                        continue
                    else:
                        raise Exception(
                            f"Cannot reduce message context Any further: {str(e)}"
                        )

                raise

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=None,
        max_value=150,
        max_time=1200,  # Maximum backoff time in seconds
        jitter=backoff.full_jitter,  # Add jitter to avoid thundering herd
        giveup=lambda e: is_token_limit_error(
            str(e).lower()
        ),  # Use the shared function
    )
    async def _retryable_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ChatCompletion:
        try:
            return await self.chat(messages, tools)
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            raise

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] = []
    ) -> ChatCompletion|Message:
        if self.provider == "anthropic":
            await asyncio.sleep(10)
            if self.model_name == "claude-3-7-sonnet-20250219-thinking":
                model_name = "claude-3-7-sonnet-20250219"
                if len(tools) > 0:
                    return await self.anthropic_client.messages.create(
                        model=model_name,
                        messages=messages,
                        temperature=1,
                        tools=tools,
                        thinking={"type": "enabled", "budget_tokens": 13384},
                        max_tokens=16384,
                    )
                else:
                    return await self.anthropic_client.messages.create(
                        model=model_name,
                        messages=messages,
                        temperature=1,
                        max_tokens=16384,
                        thinking={"type": "enabled", "budget_tokens": 13384},
                    )
            else:
                if len(tools) > 0:
                    return await self.anthropic_client.messages.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        tools=tools,
                        max_tokens=8192,
                    )
                else:
                    return await self.anthropic_client.messages.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=8192,
                    )

        if self.model_name in [
            "o3-mini-2025-01-31",
            "o1-2024-12-17",
            "o3-2025-04-16",
            "o4-mini-2025-04-16",
            "gpt-4.1",
            "o3"
        ]:
            if len(tools) > 0:
                return await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                )
            else:
                return await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )

        if self.model_name == "grok-3-mini-fast-beta-high-reasoning":
            if len(tools) > 0:
                return await self.client.chat.completions.create(
                    model="grok-3-mini-fast-beta",
                    messages=messages,
                    temperature=self.temperature,
                    tools=tools,
                    max_tokens=self.max_tokens,
                    reasoning_effort="high",
                )
            else:
                return await self.client.chat.completions.create(
                    model="grok-3-mini-fast-beta",
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort="high",
                )

        if self.model_name == "grok-3-mini-fast-beta-low-reasoning":
            if len(tools) > 0:
                return await self.client.chat.completions.create(
                    model="grok-3-mini-fast-beta",
                    messages=messages,
                    temperature=self.temperature,
                    tools=tools,
                    max_tokens=self.max_tokens,
                    reasoning_effort="low",
                )
            else:
                return await self.client.chat.completions.create(
                    model="grok-3-mini-fast-beta",
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort="low",
                )

        if len(tools) > 0:
            return await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                tools=tools,
                max_tokens=self.max_tokens,
            )
        else:
            return await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

    def get_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        tools = []

        # Handle Anthropic response format
        if self.provider == "anthropic":
            for content in response.content:
                if content.type == "tool_use":
                    tools.append(
                        {
                            "name": content.name,
                            "arguments": content.input,
                            "tool_content": content,
                        }
                    )
            return tools

        # Handle OpenAI response format
        for choice in response.choices:
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    tools.append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                            "tool_content": tool_call,
                        }
                    )
        return tools

    def parse_response(self, response: dict[str, Any]) -> str:
        # Handle Anthropic response format
        if self.provider == "anthropic":
            for content in response.content:
                if content.type == "text":
                    return content.text
            return ""  # Return empty string if no text content found

        # Handle OpenAI response format
        return response.choices[0].message.content

    def append_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_content: Any,
        tool_result: str,
    ) -> list[dict[str, Any]]:
        # Handle Anthropic response format
        if self.provider == "anthropic":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_content.id,
                            "content": tool_result,
                        }
                    ],
                }
            )
        # Handle OpenAI response format
        else:
            messages.append(
                {
                    "role": "tool",
                    "content": tool_result if tool_result is not None else "",
                    "tool_call_id": tool_content.id,
                }
            )
        return messages

    def convert_usage(self, usage: dict[str, Any]) -> dict[str, Any]:
        if not usage:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        if self.provider == "anthropic":
            return {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": usage.input_tokens + usage.output_tokens,
            }
        elif self.provider == 'openrouter':
            return {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.prompt_tokens + usage.completion_tokens,
            }
        else:
            return {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
