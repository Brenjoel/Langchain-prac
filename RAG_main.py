import os 

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama , OllamaEmbeddings
from langchain_pinecone  import PineconeVectorStore

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from operator import itemgetter

load_dotenv()

print("Initializing components...")


Ollama_embedding_model = "nomic-embed-text"
llm_model = "qwen3:0.6b"
embeddings = OllamaEmbeddings(model = Ollama_embedding_model)
llm = ChatOllama(model = llm_model)

vector_store = PineconeVectorStore(
    index_name = os.environ["INDEX_NAME"],
    embedding=embeddings
 )

retriever = vector_store.as_retriever(search_args = {"k":3})

prompt_template =  ChatPromptTemplate.from_template(
    """Answer the question based on the following context 
    {context}

    Question: {question}

    Provide a detailed answer:
    """
)

def format_docs(docs):
    """Format recieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

# Implementation 1: Without LCEL
def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without lcel
    Manually retrieves documents, formats them, and generates a response.

    Limitaions:
    - Manual step-bt-step execution
    - No built-in streamin support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """

    # Step 1: Retrieve relavent documents
    docs = retriever.invoke(query)

    # Step2: Format documents into context string
    context = format_docs(docs)

    # Step3: Format the prompt with context and query 
    messages = prompt_template.format_messages(context = context , question = query)

    # step4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    return response.content

# Implementaion 2 Implementaion with LCEL ( Better appraoch)
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language)
    return a chain that can be invoked with {"question":"..."}
    
    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator(|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with langchain's type system
    - Less code: More concise and readable
    - Better debugging: LangChain provides better observability tools

    """
    retrieval_chain = (
        RunnablePassthrough.assign(context = itemgetter('question') | retriever | format_docs)
    | prompt_template
    | llm
    | StrOutputParser()
    )

    return retrieval_chain

if __name__ == "__main__":
    print("Retrieving..")

    query = "What is pinacone in machine learning"
    
    #Option 0 : RAW invokation without RAG
    print("\n")
    print("="*70)
    print("Implementaion 0: RAW LLM invokation (NO RAG)")
    print("="*70)
    
    # result_raw = llm.invoke([HumanMessage(content = query)])
    # print("\nAnswer: \n",result_raw.content)

    # Option 1: RAW invokation without LCEL
    print("\n")
    print("="*70)
    print("Implementaion 1: RAW LLM invokation (Without LCEL)")
    print("="*70)
    
    # result_without_lcel = retrieval_chain_without_lcel(query)
    # print("\nAnswer: \n",result_without_lcel)

    
    # Option 2: Use implemtation with LCEL (Better appraoch)
    print("\n")
    print("="*70)
    print("Implementaion 2:  LLM invokation (With LCEL)")
    print("="*70)
    
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question":query})
    print("\nAnswer: \n",result_with_lcel)
