# from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import os
# from graph.chains.retrieval_grader import llm

MODEL="llama-3.3-70b-versatile"

llm = ChatGroq(temperature=0,model=MODEL,api_key=os.environ["GROQ_API_KEY"])


# prompt1 = hub.pull("rlm/rag-prompt")
# from langsmith import Client

# client = Client()

# prompt = client.pull_prompt("rlm/rag-prompt",dangerously_pull_public_prompt=True)

# print(type(prompt))
# print(prompt)
# print(prompt.input_variables)


# from langsmith import Client

# # client = Client()
# # prompt = client.pull_prompt("rlm/rag-prompt",dangerously_pull_public_prompt=True)


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
)

generation_Chain = prompt | llm | StrOutputParser()
