from typing import Any , Dict
from graph.chains.generation import generation_Chain
from graph.state import GraphState

def generate(state: GraphState) -> Dict[str,Any]:
    print("---Generate---")
    question = state["question"]
    documents = state["document"]

    generation = generation_Chain.invoke({"context":documents,"question":question})
    return {"documents":documents, "question":question,"generation":generation}
