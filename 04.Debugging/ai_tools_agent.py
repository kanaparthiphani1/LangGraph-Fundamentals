from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_groq import ChatGroq
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


class State(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

model = ChatGroq(model="qwen/qwen3-32b")

def make_tools_graph():

    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b
    
    api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
    arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
    
    api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

    tools = [add,arxiv,wiki]

    tool_node = ToolNode(tools)
    model_with_tools = model.bind_functions(tools)

    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}
    
    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", tools_condition)

    agent = graph_workflow.compile()
    return agent

agent = make_tools_graph()

