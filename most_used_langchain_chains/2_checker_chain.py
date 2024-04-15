from langchain.chains.llm_checker.base import LLMCheckerChain
from langchain_openai import OpenAI

llm = OpenAI(temperature=0.5)
checker_chain = LLMCheckerChain.from_llm(llm=llm)

letter = """
The cat is on the table. The man took the cat and put it in the box. The cat is still on the table.
"""

res = checker_chain.invoke(letter)

print(res)