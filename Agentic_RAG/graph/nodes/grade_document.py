from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

def grade_d():
    pass

def grade_document(state:GraphState) -> Dict[str,Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    if any document is not relavent, we wll set a flag to run the web search

    Args:
        state (dict): The current graph state
    
    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state 
    """

    print("---Check Document Relevance to Question---")
    print("PRANK","-"*60,"\n",state.keys())
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents :
        score = retrieval_grader.invoke(
            {"question": question, "document":d.page_content}
        )

        grade = score.binary_score
        if grade.lower() == "yes":
            print("---Grade: Document Relevant---")
            filtered_docs.append(d)
        else:
            print("---GRADE: Document Not Relevant---")
            web_search = True
            continue
        
    return{"documents": filtered_docs,"question":question,"web_search":web_search}
