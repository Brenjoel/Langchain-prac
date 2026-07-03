from dotenv import load_dotenv
import os

load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

MODEL="llama-3.3-70b-versatile"

llm = ChatGroq(temperature=0,model=MODEL,api_key=os.environ["GROQ_API_KEY"])

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents"""

    binary_score : str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments) 
# Check with structured out put implementation (F12)

system = """
    You are a grader assessing relevance of a retrieved document to a user question. \n
    if the document contains keyword(s) or semantic meaning related to the question, grade it as relevence \n
    Give a binary score 'yes' or 'no score to indicate whether the document is relevant to the question .
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human", "Rtrieved document: \n\n {document} \n\n Usr question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader