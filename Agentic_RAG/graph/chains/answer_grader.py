from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq

# from graph.chains.generation import llm
import os
MODEL= 'qwen/qwen3-32b' #"llama-3.3-70b-versatile"

llm = ChatGroq(temperature=0,model=MODEL,api_key=os.environ["GROQ_API_KEY"])

class GradeAnswer(BaseModel):
    """Binary score for Answer relatability in generation answer"""

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no' "
    )

structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """ You are a grader assessing whether the answer addresses/ resolves a question.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question"""

answer_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","User question: \n\n {question} \n\n LLM generation: {generation}")
    ]
)

answer_grader : RunnableSequence = answer_grader_prompt | structured_llm_grader
