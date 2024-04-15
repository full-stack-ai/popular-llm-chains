from langchain.chains.sequential import SimpleSequentialChain

from langchain.chains.llm import LLMChain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

llm = OpenAI(temperature=0)

wiki_template = """You are a Wikipedia expert.
                    You answer common knowledge questions based on Wikipedia knowledge.
                    Your explanations are detailed and in plain English.
                    ---
                    Here is a question:
                    {question}"""

wiki_prompt = PromptTemplate(
    input_variables=["question"],
    template = wiki_template
)

wiki_chain = LLMChain(llm=llm, prompt=wiki_prompt)


verifier_template = """
You are a verifier of questions about wikipedia contents, you are tasked
to inspect the quality of answers returned from wikipedia. 
If they consist of misinformation or hallucination you should flag those. 
Your response should be only one word either the information provided by the wiki_chain is correct or not.

Here is the wiki information along with the question submitted to you:
{question}"""

verifier_prompt = PromptTemplate(
    input_variables=["question"],
    template = verifier_template
)

verifier_chain = LLMChain(llm=llm, prompt=verifier_prompt)

sequential_chain = SimpleSequentialChain(chains=[wiki_chain, verifier_chain], verbose=True)

review = sequential_chain.invoke("What was the last invention of Nikola Tesla?")
print(review)
