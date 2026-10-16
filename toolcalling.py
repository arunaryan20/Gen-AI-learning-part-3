from dotenv import load_dotenv

load_dotenv()
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage

from rich import print

# 1 Creating a tool


@tool
def get_text_length(text: str) -> int:
    """Returns the number of character in a given text"""  # docstring
    return len(text)


tools = {"get_text_length": get_text_length}

llm = ChatOllama(model="qwen2.5:7b")

# 2 Tool binding
llm_with_tool = llm.bind_tools([get_text_length])

message = []
prompt = input("You : ")
query = HumanMessage(prompt)

message.append(query)
# print(message)

result = llm_with_tool.invoke(message)

message.append(result)

# print(message)

if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)
    # print(message)

final_result = llm_with_tool.invoke(message)
print(final_result.content)


# result = llm.invoke(
#     "Returns the number of character in a given text : 'hello how are you' "
# )

# result = llm_with_tool.invoke(
#     "Returns the number of character in a given text :hello how are you"
# )

# print(result.tool_calls[0])

# print(get_text_length.invoke({'name': 'get_text_length', 'args': {'text': 'hello how are you'}, 'id': '89779862-0fb1-4d46-a566-4c038893151a', 'type': 'tool_call'}))


# if result.tool_calls:
#     tool_call = result.tool_calls[0]
#     tool_result = get_text_length.invoke(tool_call["args"])

#     final_response = llm.invoke(f"the length of text is {tool_result}")

#     print(final_response.content)
