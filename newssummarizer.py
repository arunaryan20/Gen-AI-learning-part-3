from dotenv import load_dotenv

load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearchResults(max_result=5)

llm = ChatOllama(model="qwen2.5:7b", temperature=0)

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template(
    """
   You are a helpful assistant

   summarize the following news into clear bullet points
   {news}
"""
)

chain = prompt | llm | parser

news_result = search_tool.run("Latest AI news of 2026")

result = chain.invoke({"news": news_result})

print(result)
