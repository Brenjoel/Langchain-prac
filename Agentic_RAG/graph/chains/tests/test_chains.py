# pytest . -s -v
# . indicates current working directory
#  -s to display from stdout    
#  -v verbose flag, that shows the test we run
import pytest 

from dotenv import load_dotenv
load_dotenv()
import os

from pprint import pprint

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever

from graph.chains.generation import generation_Chain

from graph.chains.hallucination_grader import hallucination_grader, GradeHallucinations
from graph.chains.router import question_router , RouteQuery

def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content

    res : GradeDocuments = retrieval_grader.invoke(
        {"question": question , "document":doc_text}
    )

    assert res.binary_score =='yes'



def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content

    res : GradeDocuments = retrieval_grader.invoke(
        {"question": "How to make a pizza" , "document":doc_text}
    )

    assert res.binary_score =='no'


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_Chain.invoke({"context":docs,"question":question})
    # pprint(generation)
    assert generation


def test_hallucination_grader_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_Chain.invoke({'context':docs,"question":question})
    res : GradeHallucinations = hallucination_grader.invoke(
        {'documents':docs , "generation": generation}
    )
    assert res.binary_score

def test_hallucination_grader_no() -> None:
    question = "how to make a pizza at home"
    docs = retriever.invoke(question)

    generation = generation_Chain.invoke({'context':docs,"question":question})
    res : GradeHallucinations = hallucination_grader.invoke(
        {'documents':docs , "generation": "In order to make a pizza we need some dough"}
    )
    assert not res.binary_score

def test_router_to_vectorstore()->None:
    question = 'agent memory'
    res: RouteQuery = question_router.invoke({"question":question})
    assert res.datasource == "vectorstore"

def test_router_to_websearch()->None:
    question = 'how to make pizza'
    res: RouteQuery = question_router.invoke({"question":question})
    assert res.datasource == "websearch"


def test_foo() -> None:
    assert 1==1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))