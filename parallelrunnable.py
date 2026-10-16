from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableParallel, RunnableLambda

# Components
model = ChatOllama(model="qwen2.5:7b", temperature=1)
parser = StrOutputParser()

# Two different Prompts
short_prompt = ChatPromptTemplate.from_template("Explain {topic} in 1-2 line")

detailed_prompt = ChatPromptTemplate.from_template("Explain {topic} in detail")

# Input
topic = "Machine Learning"

chain = RunnableParallel(
    {
        "short": RunnableLambda(lambda x: x["short"]) | short_prompt | model | parser,
        "detailed": RunnableLambda(lambda x: x["detailed"])
        | detailed_prompt
        | model
        | parser,
    }
)

# result = chain.invoke({"topic": "Write a java program to add 2 numbers"})
result = chain.invoke(
    {"short": {"topic": "Machine Learning"}, "detailed": {"topic": "Deep Learning"}}
)

print("Short : ", result["short"])
print("Detailed : ", result["detailed"])
