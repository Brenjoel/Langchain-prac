from dotenv import load_dotenv

load_dotenv()

from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import START, END , StateGraph, MessagesState

from Reflexion_AgentX_chains import revisor, first_responder
from Reflexion_agentX_Toolexecutor import execution_tools

MAX_ITERATIONS = 2

def draft_node (state:MessagesState):
    """Draft the initial response."""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def revise_node(state: MessagesState):
    """Revise the answer based on tool results."""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages":[response]}

def event_loop(state:MessagesState) -> Literal["execution_tools", END] :
    """Determine whether to continue or end based on the iteration count."""
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state['messages']
    )

    if count_tool_visits > MAX_ITERATIONS:
        return END
    return "execution_tools"

builder = StateGraph(MessagesState)
builder.add_node("Draft",draft_node)
builder.add_node("execution_tools",execution_tools)
builder.add_node("Revise",revise_node)

builder.add_edge(START, "Draft")
builder.add_edge("Draft", "execution_tools")
builder.add_edge("execution_tools", "Revise")
builder.add_conditional_edges("Revise",event_loop,["execution_tools",END])

graph = builder.compile()
# graph.get_graph().draw_mermaid_png("Reflextion_AgentX_Graph.png")
graph.get_graph().draw_mermaid()


res = graph.invoke(
    {
        "messages": [
            {
                "role":"user",
                "content": "Write about AI Powered SOC / Autonomous SOC problem domain, list the startups that do that and raised capital "
            }
        ]
    }
)

last_message = res["messages"][-1]

if isinstance(last_message,AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"])["answer"]

print(res)

if __name__ == '__main__':
    print("Hello")