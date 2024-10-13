# hidevs_gen_ai_programs_cohort_-1

import os
# environment variable: it is use to store the confedential credits like API keys, cloud or local storage's APIs and clusters or index_names etc
from dotenv import load_dotenv
#importing QA chain
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

load_dotenv()

API_KEY= os.getenv("API_KEY")

# prefix of the prompt
prefix = "You are a helpful AI content generator, creating personalized blog posts or marketing content based on provided content. " \
         "Your response should be informative, engaging, and tailored to the audience specified. "

#examples of chain of thougths to be include in prompt

EXAMPLES =[
#mutiline query
   """ Use the following formates:
    Context: {context}
    User: {query}
    AI: {answer}
"""
]

#template for generating content
content_template = """
Generate a [length: short-form, long-form] blog post targeted at [audience], focusing on the topic of [keyword]. 
Start by addressing a common challenge or pain point faced by [audience], and introduce [product/service] as a solution. 
Emphasize the unique features of [product/service] and how it helps resolve the issue. 
Use a [tone: formal, casual, friendly, professional] style throughout the post. 
Conclude with a compelling call to action that encourages readers to [desired action: purchase, subscribe, learn more, etc.]. 
Make the content engaging, informative, and easy to understand.
"""

example_prompt = PromptTemplate(
input_variable=["content", "query", "answer"],
template=content_template
)

# SUFFIX
suffix= """
Context: {context}
User: {query}
AI: 
"""
# Final CHAT_PROMPT to combine everything
CHAT_PROMPT = PromptTemplate.from_examples(
    examples=EXAMPLES, suffix=suffix, input_variables=["context", "query"], prefix=prefix
)

# Function to create a generic LLM (could be any model that follows the LangChain interface)
def create_llm(model_name, api_key):
    if model_name == "openai":
        return OpenAI(model_name="text-davinci-003", openai_api_key=api_key, temperature=0, max_tokens=1000)
    # Add more models here as needed, e.g., Groq, Mistral AI, etc.
    else:
        raise ValueError("Model not supported")

llm = create_llm("openai", OPENAI_API_KEY) 

#load the QA chain
chain= load_qa_chain(llm, chain_type= "stuff", prompt= CHAT_PROMPT, verbose = False)

#sample document for the content

docs= [
    Documents(page_content ="Our organic skincare line features natural ingredients, free from harmful chemicals, " 
                          "ideal for health-conscious consumers.",
            metadata={}),
    Document(page_content="AI-powered marketing tools help small business owners automate and analyze their campaigns effectively.",
             metadata={})
]

#returning the response
while True:
    query= input("What is your query: \n")
    response=chain.run(input_document = docs, query= query)
    print(response)

