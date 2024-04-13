from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
prompt_template = "Tell me a {adjective} joke"
prompt = PromptTemplate(
    input_variables=["adjective"], template=prompt_template
)
llm = LLMChain(llm=OpenAI(), prompt=prompt)