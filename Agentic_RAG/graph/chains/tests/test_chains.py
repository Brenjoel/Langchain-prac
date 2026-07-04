# pytest . -s -v
# . indicates current working directory
#  -s to display from stdout    
#  -v verbose flag, that shows the test we run
import pytest 

from dotenv import load_dotenv
load_dotenv()

from pprint import pprint

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever

from graph.chains.generation import generation_Chain

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
    pprint(generation)



def test_foo() -> None:
    assert 1==1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))