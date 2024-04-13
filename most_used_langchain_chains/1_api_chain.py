from langchain.chains.api.base import APIChain
from langchain_openai import ChatOpenAI
from langchain.chains.api import news_docs
from dotenv import load_dotenv, find_dotenv
import os

api_key = os.getenv("NEWS_API_KEY")

def main():
    _ = load_dotenv(find_dotenv())

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = APIChain.from_llm_and_api_docs(llm=llm,api_docs=news_docs.NEWS_DOCS,
                                           verbose=True,
                                           headers={'X-Api-Key':api_key},
                                           limit_to_domains=["https://newsapi.org"]).with_config({"run_name":"news_api_chain"})
    res = chain.invoke("Give me some new about Las Vegas?")
    print(res)


if __name__ == "__main__": 
    main()