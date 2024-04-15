from langchain_openai import OpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.prompts import PromptTemplate
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple

llm = OpenAI()

qa_prompt = PromptTemplate(
    template="Q: {question} A:",
    input_variables=["question"],
)
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

constitutional_chain = ConstitutionalChain.from_llm(
    llm=llm,
    chain=qa_chain,
    constitutional_principles=[
        ConstitutionalPrinciple(
            critique_request="Tell if this answer is good.",
            revision_request="Return each sentence in an ordered numbered bulleted list.",
        )
    ],
)

res = constitutional_chain.invoke("What is the meaning of life?")

print(res['output'])