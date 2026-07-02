from dotenv import load_dotenv
import os

from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END , StateGraph
from langgraph.graph.message import add_messages

from Reflection_agent_chains import generate_chain, reflection_chain

load_dotenv()

class MessageGraph(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

REFLECT = "Reflect"
GENERATE = "Generate"

def generation_node(state : MessageGraph):
    ret =  {"messages": [generate_chain.invoke({"messages": state["messages"]})]}
    # print("-"*160,"\nGeneration Node: ", ret)
    return ret

def reflection_node(state: MessageGraph):
    # print(state["messages"])
    res = reflection_chain.invoke({"messages": state["messages"]})
    # print("-"*160,"\nReflection res: ",res)
    ret = {"messages": [HumanMessage(content=res.content)]}
    # print("-"*160,"\nReflection node: ",ret)
    return ret

def should_continue(state: MessageGraph):
    if len(state['messages'])>6:
        return END
    return REFLECT

builder = StateGraph(state_schema=MessageGraph)

builder.add_node(GENERATE,generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)

builder.add_conditional_edges(GENERATE,should_continue,path_map={END:END,REFLECT:REFLECT})

builder.add_edge(REFLECT,GENERATE)

graph = builder.compile()
# print(graph.get_graph().draw_mermaid(),"-"*60,"\n") # https://excalidraw.com/ > more tools > mermaid t oexcalidraw and paste the code generated in the output config: ...
graph.get_graph().print_ascii()
# graph.get_graph().draw_mermaid_png(output_file_path="Reflection_agent.png")


if __name__ == '__main__':
    print("Hello")
    content1 = """
        I am a software engineer working on AI agents, who ever is interested in building AI agents join me and we can learn together
    """
    content2= """ @LangChainAl

        newly Tool Calling feature is seriously underrated.

        After a long wait, it's here- making the implementation of agents across
        different models with function calling - super easy.

        Made a video covering their newest blog post
    """
    content3 = """
        Here is a small, fun story about a clever little mouse.Once, a tiny mouse lived in a big house. His name was Pip. Pip loved to eat sweet treats. One day, a big jar of cookies sat on the high kitchen table.Pip was too short to reach it. He looked around and saw a tall, wooden chair. He pushed the chair to the table. Then, he climbed up. He was very happy.Pip reached into the jar. He took the biggest cookie he could find. The cookie was too big to hold. He pushed it with his nose. It fell to the floor with a loud thump.Suddenly, a big cat heard the noise. The cat walked into the room. He walked very fast. Pip got very scared. He did not know what to do.The cat came closer. Pip knew he had to be brave. He saw an old box on the counter. He pushed the heavy box. It fell to the floor and made a huge crash.The cat jumped. He was very afraid of the loud sound. The cat ran away out the door. Pip let out a big breath.Pip was very proud. He learned a great lesson. Being smart is better than being big. Pip picked up his big cookie. He walked back to his little hole and ate his reward.Would you like me to read this story out loud to you, or would you like to hear a story about a different animal?
    """
    story = "Leo the lion was very proud of his golden mane. One afternoon, a playful breeze swept through the savanna. It blew a bright red bird right into Leo’s hair. The bird panicked and pulled on the lion's fur. Leo roared in frustration, trying to shake the bird loose. Instead, his roar scared a nearby giraffe, who accidentally bumped a tree. A heavy coconut fell and landed right on Leo’s head. The bird flew away safely, but Leo was left with a big bump. He learned that pride can lead to a funny fall."

    input = HumanMessage(content=story)
    print("Running...")
    response = graph.invoke(input)
    print("\n\nKeys: ",response.keys())