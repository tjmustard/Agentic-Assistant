from dotenv import load_dotenv

_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage

from langchain_openai import ChatOpenAI
#from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel
#from langchain_core.pydantic_v1 import BaseModel
from tavily import TavilyClient
import os
from langchain_core.runnables.graph import MermaidDrawMethod
from PIL import Image, ImageDraw
import io
import sqlite3
from langchain.globals import set_verbose

#set_verbose(True)

from langchain.globals import set_debug

#set_debug(True)

PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""

max_web_research_loops: int = int(os.environ.get("MAX_WEB_RESEARCH_LOOPS", "3"))
local_llm: str = os.environ.get("OLLAMA_MODEL", 'llama3.1')
#search_api: SearchAPI = SearchAPI(os.environ.get("SEARCH_API", SearchAPI.DUCKDUCKGO.value))  # Default to DUCKDUCKGO
fetch_full_page: bool = os.environ.get("FETCH_FULL_PAGE", "False").lower() in ("true", "1", "t")
ollama_base_url: str = os.environ.get("OLLAMA_SERVER", "http://localhost:11434/")
gemini_api_key: str = os.environ.get("GEMINI_API_KEY")

BREAKLINE = "========================================================="
#model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
print(BREAKLINE)
print("Using the Ollama server located here:")
print(ollama_base_url)
#model = ChatOpenAI(model=local_llm, base_url=ollama_base_url + "/v1", api_key="ollama")#, format='json')
#model = ChatOllama(model=local_llm, base_url=ollama_base_url, temperature=0.4, num_ctx=8192)#4096)#, format='json')
model = ChatGoogleGenerativeAI(
                           model="gemini-2.0-flash",
                           google_api_key=gemini_api_key,
                          )
#llm = OllamaLLM(model=local_llm, base_url=ollama_base_url)#, format='json')
#print(llm)
#result = llm.invoke("Who are you?")
#type(result)
#print(result)

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

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
    response = model.invoke(messages)
    return {"plan": response.content}

def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    try:
        content = state['content']
    except:
        content = []
    try:
        for q in queries.queries:
            response = tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
    except:
        print("No queries")
    return {"content": content}

def generation_node(state: AgentState):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
        ]
    response = model.invoke(messages)
    print(BREAKLINE)
    print(response)
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
    response = model.invoke(messages)
    return {"critique": response.content}

def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    try:
        content = state['content']
    except:
        content = []
    try:
        for q in queries.queries:
            response = tavily.search(query=q, max_results=5)
            for r in response['results']:
                content.append(r['content'])
    except:
        print("No queries")
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


thread = {"configurable": {"thread_id": "1"}}

for s in graph.stream({
                       'task': "Provide me a book report for `Code of Honor` by Alan Gratz",
                       "max_revisions": 5,
                       "revision_number": 1,
                      }, thread):
    print(s)


