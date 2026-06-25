from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv

load_dotenv()

import ollama

from langsmith import traceable

MAX_ITERATIONS = 10
# MODEL = "qwen3:0.6b"
MODEL = "qwen3:8b"



@traceable(name='tool')
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product= '{product}')")
    prices = {
        "laptop" : 1299.99,
        "headphones" : 149.95,
        "keyboard" : 89.90
    }

    return prices.get(product,0)

@traceable(name='tool')
def apply_discount(price: float , discount_tier: str) -> float :
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold"""
    print(f"    >> Execution apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {
        'bronze' : 5,
        'silver' : 12,
        'gold' : 23
    }
    discount = discount_percentages.get(discount_tier,0)
    return round(price * (1-discount/100), 2)

# Agent loop

# Difference 2 : Without @tool, we must manually define the JSON schema for each function
# This is exacly what LangChain's @tool decorator generates automatically from the function's type hints and docstring
tools_for_llm = [
    {
        "type" : "function",
        "function" : {
            "name" : "get_product_price",
            "description" : "Look up the price of a product in the catalog.",
            "parameters" :{
                "type" : "object",
                "properties" : {
                    "product" : {
                        "type" : "string",
                        "description" : " The product name, e.g. 'laptop' , 'headphones' , 'keyboard'",

                    },
                },
                "required" : ["product"]
            },
        },
    },
    {
      "type": "function",
      "function": {
        "name": "apply_discount",
        "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold",
        "parameters": {
          "type": "object",
          "required": ["price" , "discount_tier"],
          "properties": {
            "price": {"type": "number", "description": "The original price"},
            "discount_tier": {"type": "string", "description": "The discount tier: 'bronze', 'silver', or 'gold'",}
          },
        },
      },
    },
]

# Helper to trace ollama call
@traceable(name = "Ollama chat", run_type = 'llm')
def ollama_chat_traced(messages):
    return ollama.chat(model = MODEL , tools = tools_for_llm , messages = messages)


@traceable(name = "Ollama Agent Loop")
def run_agent(question: str):
    # pass
    tools_dict = {
        "get_product_price" : get_product_price,
        "apply_discount" : apply_discount,
    }


    messages = [
        {
            "role" : "system", 
            "content" : (
                "You are a helpful shopping assistant. "
                "You have access to product catalog tool and a discount tool. \n\n" \
                "STRICT RULES : You must follow these rules exactly: \n"

                "1. NEVER guess or assume any product price. "
                "You must call get_product_price first to get the real price. "
                "2. Only call apply_discount_price  AFTER you gave received a price from get_product_price."
                "Pass the exact price returned by get_product_price- do NOT pass a made-up number. \n"
                "3. NEVER calculate discounts yourself using math."
                "Always use the apply discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use - do NOT assume one"
            )
        },
        {
            "role" : "user",  # in some models the human role can be human, need to check the documentation based on the model
            "content" : question
        }
    ]


    for iteration in range (1,MAX_ITERATIONS-6):
        print(f"\n ----Iteration {iteration} ----")
        
        # Difference 5
        response = ollama_chat_traced(messages = messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls 

        # If no tool calls this is the final answer
        if not tool_calls:
            print(f"Final answer is: \n {ai_message.content}")
            return ai_message.content

        # Process only the first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        # Difference 6
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments


        print(f" [Tool Selected] {tool_name} with args {tool_args}")
        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # difference 7
        observation = tool_to_use(**tool_args) 

        print(f" [Tool result] {observation}")
        messages.append(ai_message)
        messages.append(
            {
                "role" : "tool",
                "content" : str(observation)
            }
        )

if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print ()
    question = "What is the price of laptop after applying the discount for a gold tier"
    result = run_agent(question)

