from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq

# from graph.chains.generation import llm
import os
MODEL= 'qwen/qwen3-32b' #"llama-3.3-70b-versatile"

llm = ChatGroq(temperature=0,model=MODEL,api_key=os.environ["GROQ_API_KEY"])

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer"""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no' "
    )

structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """ You are a grader assessing whether an llm generation is grounded in / supported by the set of retrieved acts.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts"""

hallucinatin_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    ]
)

hallucination_grader: RunnableSequence = hallucinatin_prompt | structured_llm_grader
