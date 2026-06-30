from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()
ollama_model = 'qwen3:8b'
@tool 
def triple(num:float) -> float:
    """
    param num: a number to be tripled
    returns: gives the Triple of the given input
    """

    return float(num) * 3

tools = [TavilySearch(maX_results=1), triple]

llm = ChatOllama(model=ollama_model , temperature=0).bind_tools(tools)
