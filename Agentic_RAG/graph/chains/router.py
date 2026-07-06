from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

from graph.chains.answer_grader import llm

class RouteQuery(BaseModel):
    """Route a user query to the most relevant data source."""

    datasource : Literal["vectorstore","websearch"] = Field(
        ...,
        description="Given a user question to route it to websearch of vectorstore"
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

system = """ You are an expert at routing a user question to a vectorstore or a websearch.
The vectorstore contains documents related to agents, prompt engineering, and advarsarial attacks.
use the vectorstore for questions on thse topics.
"""

route_prompt = ChatPromptTemplate.from_messages(
    {
        ("system",system),
        ("human","{question}"),
    }
)

question_router = route_prompt|structured_llm_router
