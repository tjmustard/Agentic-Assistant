from dotenv import load_dotenv

_ = load_dotenv()

import os
import io
import datetime
import sqlite3
import ast
import argparse
from time import sleep
from PIL import Image, ImageDraw
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_community.chat_models import ChatLiteLLM
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


PLAN = """
{company}: Business Development Market Research Plan

**Objective:** To develop a comprehensive understanding of {company} to inform business development strategies, specifically identifying potential joint venture partnership opportunities within biotech and materials (with flexibility based on findings) and assessing market entry feasibility, particularly focusing on North American and Asian markets.

**Company:** {company} ( {company website}

**Target Outcome:** A written report detailing key findings regarding {company}, including competitive analysis, product roadmap insights, recent initiatives, customer base analysis, and recommendations for partnership opportunities and market entry strategies.

**I. Information Gathering & Research Areas:**

**A. Company Overview & Strategy (Foundation):**

*   **Source:** {company} website (About Us, News, Press Releases, Publications), LinkedIn, Crunchbase, Pitchbook, OpenCorporates, annual reports (if available).
*   **Information to Gather:**
    *   **Core Business:** Clearly define {company}' core business and value proposition. What problem are they solving? What is their mission?
    *   **Technology & Innovation:** Deep dive into their core technology (sodium-ion batteries). Understand the underlying science, patents (search Espacenet, Google Patents), and technological advantages/disadvantages compared to existing battery technologies (Lithium-ion, etc.).
    *   **Management Team:** Identify key executives and their backgrounds. Look for relevant experience in biotech, materials science, or battery technology.
    *   **Financial Status:** Assess financial health. Look for funding rounds, investors, revenue figures (if public), and profitability. (Crunchbase, Pitchbook, news articles).
    *   **Strategic Goals:** Identify stated or implied strategic goals. Are they focused on R&D, manufacturing, scaling production, or market penetration?
    *   **Partnerships & Collaborations:** Identify any existing partnerships with other companies, research institutions, or government agencies.

**B. Product & Technology Roadmap (Future Focus):**

*   **Source:** {company} website (Products, Technology), patent filings, industry reports, technical publications, conference presentations, job postings.
*   **Information to Gather:**
    *   **Current Product Offerings:** Detail {company}' current product offerings (e.g., battery materials, specific battery products, etc.). Include specifications, performance data, and target applications.
    *   **Product Development Pipeline:** Identify any announced or implied future product development plans. Look for hints in press releases, presentations, or job postings. Analyze recent patent applications to identify future technology directions.
    *   **Technology Roadmap:** Understand {company}' long-term vision for their technology. Where do they see sodium-ion batteries fitting into the future energy landscape? What are the key technological hurdles they need to overcome?
    *   **Manufacturing Capabilities:** Investigate their manufacturing capabilities. Do they have their own production facilities, or are they relying on contract manufacturers? What is their production capacity?

**C. Market Analysis & Competitive Landscape (Contextual Understanding):**

*   **Source:** Industry reports (e.g., BloombergNEF, Wood Mackenzie, McKinsey), market research databases (e.g., IBISWorld, MarketResearch.com), competitor websites, news articles, trade publications.
*   **Information to Gather:**
    *   **Target Market:** Define {company}' target market. Which industries are they focusing on (e.g., energy storage, electric vehicles, grid-scale storage)?
    *   **Market Size & Growth:** Estimate the size of the sodium-ion battery market and its projected growth rate. Identify key market drivers and trends.
    *   **Competitive Landscape:** Identify {company}' key competitors. This includes both direct competitors in the sodium-ion battery space and indirect competitors in the broader battery market (e.g., Lithium-ion battery manufacturers).
    *   **Competitive Analysis:** Conduct a detailed competitive analysis, comparing {company}' products, technology, pricing, and market share with those of its competitors. Identify {company}' competitive advantages and disadvantages.
    *   **Market Share:** Determine {company}' current market share (if available) and its potential to gain market share in the future.
    *   **Regulatory Landscape:** Understand the regulatory environment for battery technologies in key markets (e.g., North America, Europe, Asia). Identify any relevant regulations or incentives that could impact {company}' business.

**D. Customer Base & Relationships (Real-World Impact):**

*   **Source:** {company} website (Case Studies, Testimonials), press releases, industry events, LinkedIn, customer reviews (if available).
*   **Information to Gather:**
    *   **Customer Profile:** Define {company}' ideal customer profile. What are their needs and pain points?
    *   **Key Accounts:** Identify any key accounts or major customers that {company} is working with.
    *   **Customer Demographics:** Analyze the demographics of {company}' customer base (e.g., industry, company size, geographic location).
    *   **Customer Satisfaction:** Assess customer satisfaction levels. Look for testimonials, reviews, or case studies that provide insights into customer experiences.
    *   **Sales & Marketing Strategy:** Understand {company}' sales and marketing strategy. How are they reaching their target customers?

**E. Recent Press Coverage & Initiatives (Current Momentum):**

*   **Source:** Google News, industry publications, {company} website (News, Press Releases), social media.
*   **Information to Gather:**
    *   **Key Announcements:** Identify any recent announcements regarding new products, partnerships, funding rounds, or market expansions.
    *   **Media Sentiment:** Analyze the sentiment of media coverage. Is {company} being portrayed positively or negatively?
    *   **Key Themes:** Identify any recurring themes in media coverage. What are the key messages that {company} is trying to convey?
    *   **Social Media Presence:** Analyze {company}' social media presence. How are they engaging with their audience?

**II. Actionable Steps for Agents:**

1.  **Website Deep Dive:** Thoroughly review {company}' website, paying close attention to the sections mentioned above (About Us, News, Products, Technology, etc.).
2.  **Database Research:** Utilize databases like Crunchbase, Pitchbook, and IBISWorld to gather financial information, funding history, and market data.
3.  **Patent Search:** Conduct a patent search on Espacenet and Google Patents to understand {company}' technology and innovation pipeline.
4.  **News & Media Monitoring:** Set up Google Alerts to monitor news and media coverage related to {company} and the sodium-ion battery market.
5.  **LinkedIn Exploration:** Use LinkedIn to identify key executives and employees at {company} and to research their backgrounds and connections.
6.  **Competitor Analysis:** Research {company}' key competitors and compare their products, technology, and market share.
7.  **Customer Research:** Look for testimonials, case studies, or customer reviews that provide insights into customer experiences.
8.  **Industry Report Review:** Analyze industry reports from reputable sources like BloombergNEF and Wood Mackenzie to understand the market dynamics of the sodium-ion battery market.

**III. Report Structure:**

The final report should be structured as follows:

1.  **Executive Summary:** A brief overview of the key findings and recommendations.
2.  **Company Overview:** A detailed description of {company}' core business, technology, and management team.
3.  **Product & Technology Roadmap:** An analysis of {company}' current product offerings and future development plans.
4.  **Market Analysis & Competitive Landscape:** An assessment of the sodium-ion battery market and {company}' competitive position.
5.  **Customer Base & Relationships:** An analysis of {company}' customer profile, key accounts, and customer satisfaction levels.
6.  **Recent Press Coverage & Initiatives:** A summary of recent news and media coverage related to {company}.
7.  **Partnership Opportunities:** Identification of potential joint venture partnership opportunities in biotech and materials based on the research findings.  Specific rationale for each potential partnership should be provided.
8.  **Market Entry Feasibility:** An assessment of the feasibility of entering new markets, particularly North America and Asia, including recommendations for market entry strategies.
9.  **Conclusion:** A summary of the key findings and recommendations.
10. **Appendix:** Supporting documents, such as financial statements, patent filings, and industry reports.

**IV. Key Considerations for Partnership & Market Entry:**

*   **Synergies:**  Identify potential synergies between {company} and potential partners.  Does {company}' technology complement the partner's existing capabilities?
*   **Market Access:**  How can a partnership help {company} access new markets or customer segments?
*   **Technical Expertise:**  Does a potential partner possess technical expertise that {company} lacks?
*   **Financial Resources:**  Can a partnership provide {company} with access to additional financial resources?
*   **Competitive Advantage:**  How can a partnership strengthen {company}' competitive advantage?
*   **Regulatory Environment:**  Understand the regulatory environment in potential target markets and identify any potential barriers to entry.
*   **Cultural Differences:**  Consider cultural differences when evaluating potential partnerships and market entry strategies.

"""


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

PLAN_PROMPT = """You are an expert Business Development Market Research team lead tasked with developing a \
comprehensive plan for agents to research a specific company. This plan should outline \
the key areas of investigation and the types of information to be gathered to inform business development strategies, \
specifically to identify potential joint venture partnership opportunities and assess market entry feasibility. \
When considering partnerships or market entry, the primary industries of interest are typically biotech and materials, \
though this may vary based on the specific sub-market being explored. The research should include a competitive \
analysis, comparing company's products and services with those of its competitors, and analyzing market share. \
The plan should pay particular attention to the company's product roadmap (including general information with links \
for reference), recent press coverage regarding new initiatives, and their customer base (including size, \
demographics, key accounts, and satisfaction levels). The output of this research should be a written report. \
The plan should be clear, concise, and actionable for the agents executing the research.
"""

REPORT_PROMPT = """Your task is to iteratively write an in-depth market research report based on the plan generated \
and provided to you. The report should be less than 5 pages and targeted towards business development experts. \
You will receive information from a researcher in a structured format, which you will need to organize and synthesize \
into the report, integrating it into existing sections. There are no specific formatting guidelines. \
The report should be delivered with an understanding that the information gathering and report \
writing will be an iterative process. You will need to prioritize incorporating new information and updating the \
report.

If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""

REFLECTION_PROMPT = """You are a business development expert tasked with evaluating a market assessment report about \
a specific company. This report is intended for internal use by the team. Your role is to provide direct and \
constructive critique and actionable recommendations to the user who submitted the report. These recommendations \
should be detailed and specific, covering aspects such as the report's length, depth of analysis regarding the \
company's market position, product development, and competitive landscape, writing style, and any other areas for \
improvement relevant to a business development context.
"""
#short and concise
#These recommendations \
#should be detailed and specific, covering aspects such as the report's length, depth of analysis regarding the \
#company's market position, product development, and competitive landscape, writing style, and any other areas for \
#improvement relevant to a business development context.
#"""

RESEARCH_PLAN_PROMPT = """Generate a maximum of ten search queries for a writer compiling a business development \
market research report on the company. Each query should include the company name and aim to gather company \
information, including published reports, press releases, and any press or publications referencing the company. \
Do not use search operators in the queries."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions. \
Generate a maximum of ten search queries for a writer compiling a business development \
market research report on the company. Each query should include the company name and aim to gather company \
information, including published reports, press releases, and any press or publications referencing the company. \
Do not use search operators in the queries.
"""

max_web_research_loops: int = int(os.environ.get("MAX_WEB_RESEARCH_LOOPS", "3"))
local_llm: str = os.environ.get("OLLAMA_MODEL", 'llama3.1')
#search_api: SearchAPI = SearchAPI(os.environ.get("SEARCH_API", SearchAPI.DUCKDUCKGO.value))  # Default to DUCKDUCKGO
fetch_full_page: bool = os.environ.get("FETCH_FULL_PAGE", "False").lower() in ("true", "1", "t")
ollama_base_url: str = os.environ.get("OLLAMA_SERVER", "http://localhost:11434/")
llm_base_url: str = os.environ.get("LLM_SERVER")
llm_api_key: str = os.environ.get("LLM_API_KEY")
gemini_api_key: str = os.environ.get("GEMINI_API_KEY")
brave_api_key: str = os.environ.get("BRAVE_AI_KEY")
tavily_api_key: str = os.environ.get("TAVILY_API_KEY")

KEYBOARD_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "

BREAKLINE = "========================================================="
print(BREAKLINE)
print("Using the Ollama server located here:")
print(ollama_base_url)
#model = ChatOpenAI(model=local_llm, base_url=ollama_base_url + "/v1", api_key="ollama")#, format='json')
model_tool = ChatOpenAI(model="bedrock-llama3-1-70b", base_url=llm_base_url, api_key=llm_api_key, max_tokens=8192)#, num_ctx=20000)#, temperature=0.4, format='json')
#model_long = ChatOpenAI(model="bedrock-llama3-2-3b", base_url=llm_base_url, api_key=llm_api_key, max_tokens=40000)
#model_tool = ChatOllama(model="bedrock-llama3-2-3b", base_url=llm_base_url, api_key=llm_api_key)#, num_ctx=20000)#, temperature=0.4, format='json')
#model_tool = ChatOllama(model="llama3.1", base_url=ollama_base_url)#, num_ctx=20000)#, temperature=0.4, format='json')
#model_tool = ChatOllama(model="llama3.1", base_url=ollama_base_url, temperature=0.4, num_ctx=20000)#, format='json')
#model_tool = ChatOllama(model="mistral", base_url=ollama_base_url, temperature=0)#, num_ctx=40000)#, format='json')
#model_tool = ChatOllama(model="qwen2.5", base_url=ollama_base_url, temperature=0, num_ctx=40000)#, format='json')
#model_tool = ChatOllama(model="llama3.1", base_url=ollama_base_url, temperature=0, num_ctx=40000)#, format='json')
#model_tool = ChatOllama(model="llama3.2", base_url=ollama_base_url, temperature=0, num_ctx=80000)#, format='json')
#model = ChatOllama(model="gemma3", base_url=ollama_base_url)#, temperature=0.4, num_ctx=4096)#, format='json')

model_long = ChatGoogleGenerativeAI(
                           model="gemini-2.0-flash",
                           google_api_key=gemini_api_key,
                          )
model_tool = ChatGoogleGenerativeAI(
                           model="gemini-2.0-flash",
                           google_api_key=gemini_api_key,
                          )

tavily = TavilyClient(api_key=tavily_api_key)
brave = BraveSearch.from_api_key(api_key=brave_api_key, search_kwargs={"count": 5})

#loader = WebBaseLoader("https://www.example.com/")

db_path = 'checkpoints.db'
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)

def print_and_save_multilevel_dict(dictionary, filename="agent.log", indent=0):
    """
    Prints and saves the contents of a multi-level dictionary.

    Args:
        dictionary (dict): The dictionary to print and save.
        filename (str): The name of the output file. Defaults to "output.txt".
        indent (int, optional): The indentation level for printing. Defaults to 0.
    """
    if os.path.exists(filename):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    with open(filename, append_write) as f:
        def _print_and_save_multilevel_dict(dictionary, indent):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    # If the value is another dictionary, recursively call this function
                    print(f"{' ' * indent}{key}:")
                    print(f"{' ' * (indent + 4)}{{", file=f)
                    _print_and_save_multilevel_dict(value, indent + 4)
                    print(f"}}\n", file=f) # note: added newline at the end
                else:
                    # Otherwise, just print out the key-value pair and save it to the file
                    print(f"{' ' * indent}{key} = {value}")
                    f.write(f"{key} = {value}\n")  # write to file

        _print_and_save_multilevel_dict(dictionary, indent)

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
            content=REPORT_PROMPT.format(content=content)
        ),
        user_message
        ]
    response = model_long.invoke(messages)
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
    if "content" in state:
        content = state['content']
    else:
        content = []

    try:
        queries = model_tool.with_structured_output(Queries).invoke([
            SystemMessage(content=RESEARCH_PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ])

        for q in queries.queries:
            response = brave.run(str(q))
            sleep(1)
            for r in ast.literal_eval(response):
                cleaned_snippet = ''.join(c for c in r['snippet'] if c in KEYBOARD_CHARS)
                content.append(cleaned_snippet)  # Brave
    except:
        print(BREAKLINE)
        print("Search Failed")
        print(BREAKLINE)

    return {"content": content}

def research_critique_node(state: AgentState):
    if "content" in state:
        content = state['content']
    else:
        content = []

    try:
        queries = model_tool.with_structured_output(Queries).invoke([
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT), #RESEARCH_PLAN_PROMPT), #
            HumanMessage(content=state['critique'])
        ])

        for q in queries.queries:
            response = brave.run(str(q))
            sleep(1)
            for r in ast.literal_eval(response):
                cleaned_snippet = ''.join(c for c in r['snippet'] if c in KEYBOARD_CHARS)
                content.append(cleaned_snippet)  # Brave
    except:
        print(BREAKLINE)
        print("Search Failed")
        print(BREAKLINE)

    return {"content": content}

def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"


def get_args():
    """Define and parse arguments using argparse."""

    parser = argparse.ArgumentParser(description='Example Argument Parser')

    # Define a positional argument (not required) and two optional arguments
    parser.add_argument('--company', type=str, required=True, help='Company name')
    parser.add_argument('--website', type=str, required=True, help='Website URL')
    parser.add_argument('--maxrev', type=int, default=10, help='Maximum revisions.')

    return parser.parse_args()


def main():
    """The main execution path."""

    args = get_args()

    if hasattr(args, 'company'):
        print(f'Company: {args.company}')

    # If you pass a company but not a website it will display None
    if hasattr(args, 'website'):
        print(f"Website: {args.website}")

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

    ## Create an image of the Agent network
    #img = graph.get_graph().draw_png()
    #image = Image.open(io.BytesIO(img))
    #image.save("graph.png")
    #
    ### Create another graphic of the network
    #img2 = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    #image2 = Image.open(io.BytesIO(img2))
    #image2.save("graph-mermaid.png")

    today = datetime.date.today()
    thread_id = today.strftime("%Y%m%d")
    thread = {"configurable": {"thread_id": thread_id}}

    company = args.company
    website = args.website
    max_revisions = args.maxrev
    logfile = "{}.log".format(company)
    report_num = 0

    for s in graph.stream({
        'task': "Look into the company named {} with website {}".format(company, website),
        "max_revisions": max_revisions,
        "revision_number": 1,
    }, thread):
        print_and_save_multilevel_dict(s, filename=logfile)

        #Now save the report to the next report file
        if "generate" in s:
            report_num += 1
            filename = "{}-report-{}.txt".format(company, report_num)
            with open(filename, 'w') as output_file:
                output_file.write(BREAKLINE + '\n')
                output_file.write('Revision number: {} \n'.format(report_num))
                output_file.write('\n')
                output_file.write(s['generate']['draft'])


if __name__ == '__main__':
    main()
