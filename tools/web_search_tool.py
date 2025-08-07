import os
from langchain.tools import Tool
from langchain_tavily import TavilySearch
def build_web_search_tool() -> Tool:
    """
    Builds the Tavily web search tool.
    This tool requires the TAVILY_API_KEY environment variable to be set.
    """
    if not os.getenv("TAVILY_API_KEY"):
        raise ValueError("TAVILY_API_KEY environment variable not set. Please add it to your.env file.")

    # Tavily is optimized for AI agents and is a robust choice.
    # max_results can be adjusted as needed.
    search = TavilySearch(max_results=3)
    return Tool(
        name="WebSearchTool",
        func=search.invoke,
        description="Searches the web to retrieve missing context or answer questions about recent events."
    )