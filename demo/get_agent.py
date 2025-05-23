import traceback

from demo.agent import Agent
from demo.llm import GeneralLLM
from demo.tools import GoogleWebSearch, ParseHtmlPage, RetrieveInformation,ReadWebpageContent, QuickGoogleSearch


async def get_agent(model_name: str, parameters: dict, *args, **kwargs):
    max_turns = parameters.get("max_turns", 50)
    available_tools = {
        "quick_search_using_google": QuickGoogleSearch,
        "retrieve_information": RetrieveInformation,
        "read_webpage_content_by_urls": ReadWebpageContent,
    }

    selected_tools = {}
    for tool in parameters.get("tools", available_tools.keys()):
        if tool not in available_tools:
            raise Exception(
                f"Tool {tool} not found in tools. Available tools: {available_tools.keys()}"
            )
        selected_tools[tool] = available_tools[tool]()

    provider, model_key = model_name.split("/", 1)
    llm = GeneralLLM(
        provider=provider,
        model_name=model_key,
        max_tokens=parameters.get("max_output_tokens", 16384),
        temperature=parameters.get("temperature", 0.0),
    )

    agent = Agent(llm=llm, tools=selected_tools, max_turns=max_turns)

    return agent
