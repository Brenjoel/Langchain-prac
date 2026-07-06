from langchain_core.prompts import PromptTemplate
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

from hello_world_llm_main import AgentResponse
import re
from operator import itemgetter

def test():
    information = """
    Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and former public official known for his leadership of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025; as of June 2026, Forbes estimates his net worth to be US$835 billion.

Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.

In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research, but later left; growing discontent with the organization's direction and leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired the social network Twitter, implementing significant changes, and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017. In November 2025, Tesla approved a pay package worth $1 trillion for Musk, which he is to receive over 10 years if he meets specific goals.

Musk is a supporter of global far-right politics, figures, and political parties. He was the largest donor in the 2024 U.S. presidential election, where he supported Donald Trump. After Trump was inaugurated as president in January 2025, Musk served as Senior Advisor to the President and as the de facto head of the Department of Government Efficiency (DOGE). Shortly before a public feud with Trump, Musk left the Trump administration in May 2025 and returned to managing his companies.

Musk's political activities, statements and views have made him a polarizing figure. He has been criticized for making unscientific and misleading statements, including spreading COVID-19 misinformation, promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service, following his pledge to decrease censorship. His role in the second Trump administration attracted public backlash, particularly in response to DOGE.
    """

    summary_template = """
    Given the information {information} about a person i want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables = ["information"] , template=summary_template
    )

    # llm = ChatGoogleGenerativeAI(
    #     model = "gemini-2.0-flash", 
    #     google_api_key = os.environ.get("GOOGLE_API_KEY")
    # )
    print("HI")
    # llm = ChatOllama(model="llama3")
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="llama3")

    response = llm.invoke("What is LangChain?")

    print(response.content)
    
    print("BYE")
    chain = summary_prompt_template | llm # lang chain expressiion language (LCEL)
    # this is a pipe connecting two components summary template and llm 
    # the output of the summary template is given as input to the llm and the out put of llm is given as output
    #  this chain creates a runnable object, which mwans we can invoke the chain variable 
    response = chain.invoke(input = {"information": information})
    print(response.content)
    print("BYE2")

    # import google.generativeai as genai
    # from dotenv import load_dotenv
    # import os

    # load_dotenv()

    # api_key = os.getenv("GOOGLE_API_KEY")

    # print("API KEY:", api_key)

    # genai.configure(api_key=api_key)

    # model = genai.GenerativeModel("gemini-2.0-flash")

    # response = model.generate_content("Hello")

    # print(response.text)

def test2():    

    print("Hello from langchain-Project!")
    model = "qwen3:8b"
    model2="llama3.1:8b"
    llm = ChatOllama(model=model)  # qwen3:8b
    tools = [TavilySearch()]
    agent = create_agent(model=llm , tools = tools, response_format=AgentResponse)
    # content = "Search for 3 job postings for AI engineer using langchain in Hyderabad on linkedin and list their details"
    content = "What is langchain, explain with exxample"
    result = agent.invoke({"messages": HumanMessage(content=content)})
    # print(result)
    # print("Done")
    # print(result.keys())

    response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is Python?"
            }
        ]
    }
)

    print(response["messages"][-1].content)
    print(result)
    print(type(result))

def various_llms():
    
    print("Hello from langchain-Project!")
    # model_groq="llama-3.3-70b-versatile"
    model_gemini="gemini-2.5-flash" # "gemini-2.0-flash"
    model_gemini2="gemini-2.5-flash-preview"
    model= "qwen3:8b"
    model2="llama3.1:8b"
    # llm = ChatGroq(model=model_gemini)
    # llm = ChatGoogleGenerativeAI(model=model_gemini)
    llm = ChatOllama(model=model)  # qwen3:8b
    tools = [TavilySearch()]
    agent = create_agent(model=llm , tools = tools, response_format=AgentResponse)
    content = "Search for 3 job postings for AI engineer using langchain in Hyderabad on linkedin and list their details"
    # content = "What is the temperature in tokyo"
    result = agent.invoke({"messages": HumanMessage(content=content)})
    print('-------------------------------------------------------------------------'*2 +'\n\n')
    print(result)
    print('-------------------------------------------------------------------------'*2 +'\n\n')
    print(type(result))
    # for i, msg in enumerate(result["messages"]):
    #     print(f"\n--- Message {i} ---")
    #     print(type(msg))
    #     print(msg)

# print("HELLO")
def prompt():
    messages = [
        SystemMessage(
            content=(
            "You are a helpful shopping assistant. "
            "You have access to a product catalog tool "
            "and a discount tool. \n\n"
            "STRICT RULES - you must follow these exactly:\n"
            "1. NEVER guess or assume any product price. "
            "You MUST call get_product_price first to get the real price. \n"
            "2. Only call apply_discount AFTER you have received "
            "a price from get_product_price. Pass the exact price "
            "returned by get_product_price - do NOT pass a made-up number. \n"
            "3. NEVER calculate discounts yourself using math. "
            "Always use the apply_discount tool. \n"
            "4. If the user does not specify a discount tier, "
            "ask them which tier to use - do NOT assume one."

            )
        ),
    ]

def item_getter():
    d={"q":17,'w':23}
    a = itemgetter('q','w')
    print(d['q'])
    print(a(d))

# item_getter()s
from langchain_ollama import OllamaEmbeddings

# print(OllamaEmbeddings.model_fields.keys())

def test_embedding_batch_documents():
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
    from langchain_pinecone import PineconeVectorStore
    from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

    from logger import (Colors , log_error, log_header ,log_info,log_success,log_warning)


    load_dotenv()
    Ollama_embedding_model = "nomic-embed-text"


    # Configure SSL context to use Certifi certifications
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

    embeddings = OllamaEmbeddings(
        model = Ollama_embedding_model
    )
    # For openAI models you can use the following parameters , chunk_size = 50, retry_min_seconds=10, show_progress_bar= True

    # chroma = Chroma(persist_directly = "chroma_db" , embedding_function= embeddings)
    vector_store = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
    tavily_extract = TavilyExtract()
    tavily_map = TavilyMap(max_depth = 5, max_breadth = 20 , max_pages = 1000)
    tavily_crawl = TavilyCrawl()


    async def index_documents_async(documents : List[Document], batch_size: int = 50):
        """ Process documents in batches asyncronously"""
        
        # Create batches
        batches = [
            documents[i: i+ batch_size] for i in range(0,len(documents), batch_size)
        ]

        async def add_batch(batch: List[Document], batch_num: int)  :
            try: 
                await vector_store.aadd_documents(batch)
                
            except Exception as e:
                return False
            return True
        
        #  Process batches concurrently
        tasks = [add_batch(batch,i+1) for i,batch in enumerate(batches) ]
        results = await  asyncio.gather(*tasks , return_exceptions=True)

        successful = sum(1 for result in results if result is True)


        print("Done")

    async def main():
        """Main async function to orchestarates the entire process"""

        res = tavily_crawl.invoke(
            {
                "url" : "https://python.langchain.com/",
                "max_depth":2,
                "extract_depth": "advanced",
                "instructions":"ALL" # give any insrtruction to get the results based on the instructions Ex: "content on AI agents"
            }
        )

        all_docs = [Document(page_content=result['raw_content'], metadata= {"source":result['url']}) for result in res['results'] ]
        
        text_splitters = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splitted_docs = text_splitters.split_documents(all_docs)
        await index_documents_async(splitted_docs,batch_size=500)

    if __name__ == "__main__":
        asyncio.run(main())

def regex():
    output = "Action: Please go right and then go left"
    action_match = re.search(r"Action:\s*(.+)", output)
    tool_name = action_match.group(1).strip()
    print(action_match,"\n-------------\n",tool_name)

def groq():
    from langchain_groq import ChatGroq
    import os

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        api_key=os.environ["GROQ_API_KEY"],
    )
    print(llm.invoke("Hi").content)
groq()
