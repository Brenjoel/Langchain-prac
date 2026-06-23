import os
import ssl 

import asyncio
from typing import Any , Dict , List

import certifi
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeSparseVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap


load_dotenv()