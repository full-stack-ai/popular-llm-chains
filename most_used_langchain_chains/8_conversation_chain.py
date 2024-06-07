from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

llm = OpenAI(temperature=0)

template = """
    The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    Current conversation:
    {history}
    Friend: {input}
    AI Assistant:
"""
prompt = PromptTemplate(input_variables=["history", "input"], template=template)

conversation = ConversationChain(
    prompt=prompt,
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(ai_prefix="AI Assistant", user_prefix="Friend"),
)
conversation.predict(input="Hi There!")
print(conversation.predict(input="What's the weather like today in Edmonton?"))