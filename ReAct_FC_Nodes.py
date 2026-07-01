from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from ReAct_FC_react import llm, tools

SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
Strictly use the tools whenever required
- You have tools that finds temperature and a tool that triples the temperature
- do not hallusinate
- Analyse the tool you have and then act accordingly

"""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node
    """
    response = llm.invoke([{"role":"system","content":SYSTEM_MESSAGE}, *state["messages"]])
    return {"messages": [response]}

tool_node = ToolNode(tools)
