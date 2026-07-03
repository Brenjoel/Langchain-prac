from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeEmbeddings
from langchain_ollama import OllamaEmbeddings

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
"https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
"https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

docs = [WebBaseLoader(url).load() for url in urls]
doc_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = 250, chunk_overlap=0)
doc_splits = text_splitter.split_documents(doc_list)

# Index only once
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="RAG_Chroma",
#     embedding=OllamaEmbeddings(model="nomic-text-embed"),
#     persist_directory="./.chroma",

# )

retriever = Chroma(
    collection_name="RAG_Chroma"
).as_retriever()

pass