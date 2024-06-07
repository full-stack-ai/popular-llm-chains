from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory, ConversationKGMemory
from langchain_core.prompts.prompt import PromptTemplate
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

memory = ConversationBufferMemory(ai_prefix="AI Assistant", user_prefix="Friend")
# memory = ConversationSummaryMemory(ai_prefix="AI Assistant", user_prefix="Friend")
# memory = ConversationBufferWindowMemory(ai_prefix="AI Assistant", user_prefix="Friend", k=4, memory_key="chat_history")

conversation = ConversationChain(
    prompt=prompt,
    llm=llm,
    verbose=True,
    memory=memory,
)
conversation.predict(input="Hi There!")
conversation.predict(input="Who will win the Stanley Cup in 2024? Oilers or Panthers?")
conversation.predict(input="I want to know how self-attention impacts the performance of transformer models.")
conversation.predict(input="Are there any optimization techniques for these models?")
conversation.predict(input="Can you write a c code to develop a transformer model?")
print(conversation.predict(input="Do you know how I started my conversation with you?"))
