from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

ollama_model = "qwen3:0.6b"
# ollama_model = "qwen3:8b"

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twittwer influencergrading a tweet. Generate critique and recocmendations for the user's tweet ."
            "Always povide detailed recomendation, including requests for length, virality ,style, etc."
            "Also you have freedom to add and modify content that gives better insights"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writting excellent twitter posts."
            "Generate the best twitter post possible for the user's request"
            "If the user provides critique, respond with a revised version of your prevous attempts."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOllama(model = ollama_model)
generate_chain = generation_prompt| llm
reflection_chain = reflection_prompt | llm

# print("\n\n",reflection_chain.invoke({"messages":"Hi"}))