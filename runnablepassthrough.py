from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama

model = ChatOllama(model="qwen2.5:7b",temperature=0)

parser=StrOutputParser()

code_prompt= ChatPromptTemplate.from_messages([
    ("system","you are a code generator"),
    ("human","{topic}")
])

explain_prompt= ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant who explain code in simple terms"),
        ("human","explain the following code in the simple words:\n{code}")
    ]
)

seq= code_prompt | model | parser 

seq2 =RunnableParallel(
    {
        "code":RunnablePassthrough(),
        "explanation": explain_prompt | model | parser
    }
)

chain =seq | seq2



result= chain.invoke({"topic":"Write the code to check the prime number in java"})

print(result['code'])

print(result['explanation'])