from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import END , StateGraph

from graph.consts import RETRIEVE, GRADE_DOCUMENTS,GENERATE,WEBSEARCH
from graph.nodes import generate,retrieve,web_search,grade_document
from graph.state import GraphState

def decide_to_generate(state:GraphState):
    print("---Assess Graded Documents")

    if state["web_search"]:
        print("---Decision: Not all the documents are relevant to question, ")
        return WEBSEARCH
    else:
        print("---Decision: Generate---")
        return GENERATE

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE,retrieve)
workflow.add_node(GRADE_DOCUMENTS,grade_document)
workflow.add_node(GENERATE,generate)
workflow.add_node(WEBSEARCH,web_search)

workflow.set_entry_point(RETRIEVE)

workflow.add_edge(RETRIEVE,GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS,decide_to_generate,{WEBSEARCH:WEBSEARCH,GENERATE:GENERATE,},)
workflow.add_edge(WEBSEARCH,GENERATE)
workflow.add_edge(GENERATE,END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="Agent_graph.png")
print(RETRIEVE)