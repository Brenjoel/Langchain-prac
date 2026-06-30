from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()
ollama_model = 'qwen3:0.6b'
@tool 
def triple(num:float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """

    return float(num) * 3

tools = [TavilySearch(maX_results=1), triple]

llm = ChatOllama(model=ollama_model , temperature=0)
llm.bind_tools(tools)
print("Done")