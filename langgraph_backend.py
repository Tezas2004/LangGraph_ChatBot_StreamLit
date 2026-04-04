from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Get Hugging Face token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.7,
    huggingfacehub_api_token=hf_token
)

chat_model = ChatHuggingFace(llm=llm)

# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Chat node
def chat_node(state: ChatState):
    messages = state["messages"]
    response = chat_model.invoke(messages)
    return {"messages": [response]}

# Memory
checkpointer = InMemorySaver()

# Graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


