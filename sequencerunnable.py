from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# prompt template
prompt = ChatPromptTemplate.from_template("Explain {topic} in simple words")

# model
model = ChatOllama(model="qwen2.5:7b", temperature=0)

# output parser
parser = StrOutputParser()

# i am using chain here

chain=prompt | model | parser

response=chain.invoke("Machine Learning")

print(response)



# This is without chain

# # format the prompt
# formatted_prompt = prompt.format_messages(topic="What is Machine learning?")

# # call the model manually
# response = model.invoke(formatted_prompt)

# # Parse the output manually
# final_output = parser.parse(response.content)

# print(final_output)
