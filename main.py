from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field



load_dotenv()

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str = Field(description="the URL of the source")

class AgentResponse(BaseModel):
    """ Schema for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(default_factory=list, description="List of the sources used to generate the answer")




# llm = ChatOllama(
#     model="gpt-oss:20b",
#     temperature=0
# )

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",  # stable, supported
    temperature=0
)

tools = [TavilySearch()]

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse
)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="search for 3 job posting for an ai engineer using langchain in the Krakow Poland area on linkedin and list their details.")})
    print(result)





if __name__ == "__main__":
    main()
