from dotenv import load_dotenv

import requests
from langchain_huggingface import HuggingFaceEndpoint
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchainhub import hub  # <--- Use this specific import

load_dotenv()

# -------------------- TOOLS --------------------

search_tool = DuckDuckGoSearchRun()


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather of a city.

    Args:
        city: Name of the city.
    """
    url = f"https://api.weatherstack.com/current?access_key=YOUR_KEY&query={city}"
    data = requests.get(url).json()
    return f"{data['current']['temperature']}°C and {data['current']['weather_descriptions'][0]}"


tools = [search_tool, get_weather]

# -------------------- LLM --------------------

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3", temperature=0.1, max_new_tokens=512
)

# -------------------- ReAct Prompt --------------------

prompt = hub.pull("hwchase17/react")

# -------------------- ReAct Agent --------------------

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# -------------------- LOOP --------------------

while True:
    user_query = input("\nYou: ")

    if user_query.lower() in ["exit", "quit", "bye"]:
        break

    response = agent_executor.invoke({"input": user_query})
    print("\nAgent:", response["output"])
