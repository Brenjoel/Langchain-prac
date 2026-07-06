from typing import List

from pydantic import BaseModel, Field

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field (description="Critique of what is superluous")

class AnswerQuestion(BaseModel):
    """Answer the Question"""

    answer: str = Field(description="250 word detailed answer to the question")
    reflection: Reflection = Field(description="Your reflection on the initial answer")
    search_queries: List[str] = Field(
        description="1-2 search queries for reasearching improvements to address the critique of your answwer"
    )

class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references : List[str] = Field(description="Citations motivating your updated answer.")
     