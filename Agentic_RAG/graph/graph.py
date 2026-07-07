from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import END , StateGraph

from graph.consts import RETRIEVE, GRADE_DOCUMENTS,GENERATE,WEBSEARCH
from graph.nodes import generate,retrieve,web_search,grade_document
from graph.state import GraphState

from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader
from graph.chains.router import RouteQuery, question_router
from graph.consts import *

def decide_to_generate(state:GraphState):
    print("---Assess Graded Documents")

    if state["web_search"]:
        print("---Decision: Not all the documents are relevant to question, ")
        return WEBSEARCH
    else:
        print("---Decision: Generate---")
        return GENERATE

def grade_generation_in_documents_questions(state : GraphState) -> str:
    print("---Check Hallucinations")
    question = state['question']
    documents = state['documents']
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents,"generation":generation}
    )

    if hallucination_grade := score.binary_score:
        print("---Decision: Generation is Grounded in documents---")
        print("---Grade Generation vs Question---")
        score = answer_grader.invoke({"question":question,"generation":generation})
        if answer_grade := score.binary_score:
            print("---Decision: Generation Addresses Question---")
            return "useful"
        else:
            print("---Decision: Generation does NOT Addresses Question---")
            return "not useful"
    else:
        print("---Decision: Generation is NOT Grounded in documents---")
        return "not supported"
            
def route_question(state :GraphState)-> str:
    print("----Route Question--")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question":question})
    if source.datasource == "WEBSEARCH":
        print("---Route Question to Web Search---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---Route Question to RAG---")
        return RETRIEVE
    


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE,retrieve)
workflow.add_node(GRADE_DOCUMENTS,grade_document)
workflow.add_node(GENERATE,generate)
workflow.add_node(WEBSEARCH,web_search)

workflow.set_entry_point(RETRIEVE)

workflow.add_edge(RETRIEVE,GRADE_DOCUMENTS)

workflow.set_conditional_entry_point(route_question,{WEBSEARCH:WEBSEARCH,RETRIEVE:RETRIEVE})

workflow.add_conditional_edges(GRADE_DOCUMENTS,decide_to_generate,{WEBSEARCH:WEBSEARCH,GENERATE:GENERATE,},)
workflow.add_conditional_edges(GENERATE,grade_generation_in_documents_questions,{"not supported":GENERATE,"useful":END,"not useful":WEBSEARCH})

workflow.add_edge(WEBSEARCH,GENERATE)
workflow.add_edge(GENERATE,END)

app = workflow.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().draw_mermaid_png(output_file_path="Agentic_RAG/Agent_graph.png")
print(RETRIEVE)