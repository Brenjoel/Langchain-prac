from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

from Reflexion_AgentX_schema import AnswerQuestion, ReviseAnswer

taviy_tool = TavilySearch(max_results = 5)

def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries"""
    return taviy_tool.batch([{"query":query} for query in search_queries])

execution_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name = AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name = ReviseAnswer.__name__),
    ]
)