from dotenv import load_dotenv

_ = load_dotenv()

import os
import io
import datetime
import sqlite3
import ast
import re
from time import sleep
from PIL import Image, ImageDraw
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from pydantic import BaseModel
#from langchain_core.pydantic_v1 import BaseModel

#LLM Model
from langchain_openai import ChatOpenAI
#from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI

#Tools
from tavily import TavilyClient
from langchain_community.tools import BraveSearch
from langchain_community.document_loaders import WebBaseLoader

#Verbosity
from langchain.globals import set_verbose
from langchain.globals import set_debug
#set_verbose(True)
#set_debug(True)

PROMPT = """'''You are a helpful ai agent. Use the following tools(Only when you should use):

{tool_names}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid action values: "Final Answer" or {tools}


Follow this format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [duckduckgo_search, Calculator, Wikipedia]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
'''"""

PLAN_PROMPT = """Create a 7-day meal plan for dinner for a researcher and writer, designed to feed \
2 adults and 2 children (ages 6 and 12). The plan should prioritize generally nutritious meals, with an \
emphasis on high protein content, considering the current season (mid-March in Liberty Lake, Washington, USA). \
Include a Mexican-inspired meal, preferably tacos or a similar dish, on Tuesday. Also, include a pasta dish, \
typically with meatballs, on another night. Recipes should be simple and quick to prepare, ideally taking 30-45 \
minutes and no more than 1 hour. Exclude rosemary and chicken nuggets from all recipes. The total budget for \
groceries should be around $200 per week. Please include the full recipes for each meal, ensuring most recipes \
utilize common pantry staples and readily available, seasonal ingredients, while allowing for one more adventurous \
meal during the week. There are no kitchen appliance limitations."""

MEAL_PROMPT = """Write a 5-day meal plan based on the information provided by the "meal planner." \
The "meal planner" will provide all necessary information, including dietary restrictions, calorie goals, \
preferred cuisines, specific ingredients, available time for cooking, desired meal plan format \
(including whether recipes or just suggestions are needed), specific considerations (such as budget, \
ingredient availability, or portability), and the target audience.

If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""

REFLECTION_PROMPT = """You are a nutritionist reviewing a user's meal plan submission for general health. \
Provide a detailed critique and recommendations for improvement, focusing on nutrition, variety, and the user's \
preferences. The user has no dietary restrictions or allergies, dislikes rosemary and chicken nuggets, is \
tracking protein intake, and has a light to moderate activity level. Your response should include specific \
questions to gather more information about these aspects."""

RESEARCH_PLAN_PROMPT = """Generate a list of up to 10 search queries designed to find information for planning \
a healthy, family-friendly meal plan. The meal plan should cater to two adults and two children (ages 6 and 12), \
prioritizing meals that are not too spicy, incorporate more vegetables, and are quick to prepare. Dinners only.

Focus on search queries that will yield:
1. Recipes for family-friendly, vegetable-rich meals.
2. Nutritional information related to these types of meals.
3. Strategies and recipes for dealing with picky eaters, particularly regarding vegetables.
4. Recipes where vegetables are incorporated in a less noticeable way (e.g., hidden in sauces).

Examples of good search queries include (but are not limited to):
* 'easy family meals with hidden vegetables'
* 'healthy kid-friendly recipes quick'
* 'vegetable recipes for picky eaters'
* 'nutritional meal plans for families with children'

Ensure the search queries are concise, relevant, and likely to produce useful results for the meal planning task."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher tasked with providing information that can be used when making \
any requested revisions to a meal plan prompt. Generate a list of search queries (maximum 10) that will gather \
relevant information to help refine the prompt, focusing on clarity and organization, with the goal of providing \
feedback to improve the meal plan. Consider information related to nutrition and cooking. The desired output format \
is currently unknown."""

max_web_research_loops: int = int(os.environ.get("MAX_WEB_RESEARCH_LOOPS", "3"))
local_llm: str = os.environ.get("OLLAMA_MODEL", 'llama3.1')
#search_api: SearchAPI = SearchAPI(os.environ.get("SEARCH_API", SearchAPI.DUCKDUCKGO.value))  # Default to DUCKDUCKGO
fetch_full_page: bool = os.environ.get("FETCH_FULL_PAGE", "False").lower() in ("true", "1", "t")
ollama_base_url: str = os.environ.get("OLLAMA_SERVER", "http://localhost:11434/")
gemini_api_key: str = os.environ.get("GEMINI_API_KEY")
brave_api_key: str = os.environ.get("BRAVE_AI_KEY")
tavily_api_key: str = os.environ.get("TAVILY_API_KEY")

KEYBOARD_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "

def print_multilevel_dict(dictionary, indent=0):
    """
    Prints out the contents of a multi-level dictionary.

    Args:
        dictionary (dict): The dictionary to print.
        indent (int, optional): The indentation level. Defaults to 0.
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            # If the value is another dictionary, recursively call this function
            print(f"{' ' * indent}{key}:")
            print_multilevel_dict(value, indent + 4)
        else:
            # Otherwise, just print out the key-value pair
            print(f"{' ' * indent}{key} = {value}")

BREAKLINE = "========================================================="
#model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
print(BREAKLINE)
print("Using the Ollama server located here:")
print(ollama_base_url)
#model = ChatOpenAI(model=local_llm, base_url=ollama_base_url + "/v1", api_key="ollama")#, format='json')
#model_tool = ChatOllama(model="llama3.1", base_url=ollama_base_url)#, num_ctx=20000)#, temperature=0.4, format='json')
#model_tool = ChatOllama(model="llama3.1", base_url=ollama_base_url, temperature=0.4, num_ctx=20000)#, format='json')
model_tool = ChatOllama(model="llama3.2", base_url=ollama_base_url)#, temperature=0.4, num_ctx=40000)#, format='json')
model = ChatOllama(model="gemma3", base_url=ollama_base_url)#, temperature=0.4, num_ctx=4096)#, format='json')

model_long = ChatGoogleGenerativeAI(
                           model="gemini-2.0-flash",
                           google_api_key=gemini_api_key,
                          )
#llm = OllamaLLM(model=local_llm, base_url=ollama_base_url)#, format='json')
#print(llm)
#result = llm.invoke("Who are you?")
#type(result)
#print(result)

tavily = TavilyClient(api_key=tavily_api_key)
brave = BraveSearch.from_api_key(api_key=brave_api_key, search_kwargs={"count": 5})

loader = WebBaseLoader("https://www.example.com/")

#memory = SqliteSaver.from_conn_string(":memory:")
db_path = 'checkpoints.db'
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)

class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

class Queries(BaseModel):
    queries: List[str]

def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ]
    response = model_long.invoke(messages)
    return {"plan": response.content}

def generation_node(state: AgentState):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=MEAL_PROMPT.format(content=content)
        ),
        user_message
        ]
    response = model_long.invoke(messages)
    print(BREAKLINE)
    print(state.get("revision_number", 1))
    print(response.content)
    print(BREAKLINE)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1
    }

def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft'])
    ]
    response = model_long.invoke(messages)
    return {"critique": response.content}

def research_plan_node(state: AgentState):
    queries = model_tool.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    #print(BREAKLINE)
    #print(queries)
    #print(queries.queries)
    #for q in queries.queries:
    #    print(q)
    #print(BREAKLINE)

    if "content" in state:
        content = state['content']
    else:
        content = []

    for q in queries.queries:
        response = brave.run(q)
        sleep(1)
        #print(response)
        #print(type(response))
        for r in ast.literal_eval(response):
            #print(r)
            #print(type(r))
            #for key, value in r.items():
            #    print(f"{key}: {value}")
            #print(r['snippet'])
            cleaned_snippet = ''.join(c for c in r['snippet'] if c in KEYBOARD_CHARS)
            #print(cleaned_snippet)
            content.append(cleaned_snippet)  # Brave
            #content.append(r['snippet'])  # Brave
            #print(content)
        #response = tavily.search(query=q, max_results=5)
        #for r in response['results']:
        #    content.append(r['content']) #Tavily

    return {"content": content}

def research_critique_node(state: AgentState):
    queries = model_tool.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])

    if "content" in state:
        content = state['content']
    else:
        content = []

    for q in queries.queries:
        response = brave.run(q)
        sleep(1)
        for r in ast.literal_eval(response):
            cleaned_snippet = ''.join(c for c in r['snippet'] if c in KEYBOARD_CHARS)
            content.append(cleaned_snippet)  # Brave
            # content.append(r['snippet'])  # Brave
        # response = tavily.search(query=q, max_results=5)
        # for r in response['results']:
        #    content.append(r['content']) #Tavily

    return {"content": content}

def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"


builder = StateGraph(AgentState)
builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)
builder.set_entry_point("planner")

builder.add_conditional_edges(
    "generate",
    should_continue,
    {END: END, "reflect": "reflect"}
)

builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")

builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

graph = builder.compile(checkpointer=memory)


#print(graph.get_graph().draw_mermaid())
img = graph.get_graph().draw_png()
image = Image.open(io.BytesIO(img))
image.save("graph.png")
#render("dot", "png", img, output_path="graph.png")
#display(Image(graph.get_graph().draw_png()))


## Assuming 'app' is your LangGraph application
img2 = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
image2 = Image.open(io.BytesIO(img2))
image2.save("graph-mermaid.png")


#ImageDraw(Image(image))

today = datetime.date.today()
thread_id = today.strftime("%Y%m%d")
thread = {"configurable": {"thread_id": "1"}}

for s in graph.stream({
                       'task': "Provide a 5 day meal plan for a family of 2 adults and 2 children. The boys do not like vegetables in the main dish, but prefer them on the side. Stay away from spicy foods. Some meals that are preferred include: tacos, noodles and meatballs",
                       "max_revisions": 5,
                       "revision_number": 1,
                      }, thread):
    print_multilevel_dict(s)

    #for key, value in s.items():
    #    print(f"{key}: {value}")

    #print(s)


