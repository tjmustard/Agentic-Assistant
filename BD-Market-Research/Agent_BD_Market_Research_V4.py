from dotenv import load_dotenv
_ = load_dotenv()

import os
import io
import datetime
import sqlite3
import ast
from time import sleep
from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel
import json

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables import RunnableConfig

# Tools
from langchain_community.tools import BraveSearch
from langchain_community.document_loaders import WebBaseLoader

KEYBOARD_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "

# Environment setup
llm_base_url: str = os.environ.get("LLM_SERVER", "")
llm_api_key: str = os.environ.get("LLM_API_KEY", "")
gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")
brave_api_key: str = os.environ.get("BRAVE_AI_KEY", "")

# Initialize models and tools
model_tool = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=gemini_api_key,
)

model_long = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=gemini_api_key,
)

brave = BraveSearch.from_api_key(api_key=brave_api_key, search_kwargs={"count": 5})

# Database setup
db_path = 'checkpoints.db'
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)

# Report sections
REPORT_SECTIONS = [
    "executive_summary",
    "company_overview",
    "product_portfolio",
    "market_position",
    "customer_base",
    "financial_performance",
    "potential_synergies",
    "patent_portfolio",
    "key_accounts",
    "product_competitors",
    "market_entry",
    "financial_metrics",
    "additional_insights"
]

@dataclass
class Reference:
    """Class to store reference information"""
    source: str
    url: str = ""
    date: str = ""
    snippet: str = ""

@dataclass
class SectionContent:
    """Class to store section content and its references"""
    content: str = ""
    references: List[Reference] = field(default_factory=list)
    status: str = "pending"  # pending, researching, writing, reviewing, complete
    critique: str = ""

class AgentState(TypedDict):
    """Represents the state of the agent system"""
    task: str  # Company research task
    company_name: str
    website: str
    sections: Dict[str, SectionContent]
    current_section: str
    max_revisions: int
    revision_number: int

class SearchQueries(BaseModel):
    """Model for search queries"""
    queries: List[str]

# Prompts for different agents
MAIN_PLANNER_PROMPT = """You are the lead coordinator for a comprehensive market research project. Your task is to 
coordinate research and writing for specific sections of a market research report. Each section has dedicated research 
and writing agents. You need to:
1. Determine which section to work on next
2. Ensure sections are completed in a logical order
3. Track progress and dependencies between sections

The available sections are:
{sections}

Consider dependencies (e.g., Executive Summary should be last) and information flow between sections."""

SECTION_RESEARCH_PROMPT = """You are a specialized research agent focused on gathering information for the {section} 
section of a market research report about {company}. Your task is to:
1. Generate specific search queries to gather relevant information
2. Focus on finding verifiable facts and data
3. Include source URLs and dates for all information
4. Ensure information is recent and relevant
5. Look for specific metrics and quantifiable data
6. Find competitor information where relevant

Current section content: {content}
Previous critique: {critique}"""

SECTION_WRITING_PROMPT = """You are a specialized writing agent for the {section} section of a market research report. 
Your task is to:
1. Write clear, concise, and professional content
2. Incorporate all researched information with proper citations
3. Ensure logical flow and structure
4. Highlight key insights and implications
5. Maintain consistency with other sections
6. Include all references in a structured format

Research data: {research_data}
Current content: {content}
Previous critique: {critique}"""

CRITIQUE_PROMPT = """You are an expert reviewer of market research reports. Review the following section and provide 
specific, actionable feedback on:
1. Content completeness and accuracy
2. Data support and citation quality
3. Analysis depth and insights
4. Writing clarity and professionalism
5. Integration with overall report objectives
6. Areas needing more research or detail

Section: {section}
Content: {content}"""

def print_and_save_multilevel_dict(dictionary, filename="agent.log", indent=0):
    """Prints and saves the contents of a multi-level dictionary."""
    if os.path.exists(filename):
        append_write = 'a'
    else:
        append_write = 'w'

    with open(filename, append_write) as f:
        def _print_and_save_multilevel_dict(dictionary, indent):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    print(f"{' ' * indent}{key}:")
                    f.write(f"{' ' * indent}{key}:\n")
                    _print_and_save_multilevel_dict(value, indent + 4)
                else:
                    print(f"{' ' * indent}{key} = {value}")
                    f.write(f"{' ' * indent}{key} = {value}\n")

        _print_and_save_multilevel_dict(dictionary, indent)

def preserve_state(state: Dict, updates: Dict) -> Dict:
    """Helper function to preserve state while updating specific fields."""
    return {
        "task": state["task"],
        "company_name": state["company_name"],
        "website": state["website"],
        "sections": state["sections"],
        "current_section": state["current_section"],
        "max_revisions": state["max_revisions"],
        "revision_number": state["revision_number"],
        **updates
    }

def main_planner_node(state: AgentState) -> Dict:
    """Determines the next section to work on and updates the state."""
    # Debug logging
    print("\nDEBUG: State contents at start of planner:")
    print(f"State type: {type(state)}")
    print(f"State keys: {state.keys() if isinstance(state, dict) else 'Not a dict'}")
    
    messages = [
        SystemMessage(content=MAIN_PLANNER_PROMPT.format(sections="\n".join(REPORT_SECTIONS))),
        HumanMessage(content=f"Current state: {state}")
    ]
    response = model_tool.invoke(messages)
    
    # Initialize any missing sections
    for section in REPORT_SECTIONS:
        if section not in state["sections"]:
            print(f"DEBUG: Adding missing section {section}")
            state["sections"][section] = SectionContent(
                content="",
                status="pending",
                critique="",
                references=[]
            )
        
        if state["sections"][section].status == "pending":
            print(f"DEBUG: Selected section {section} for processing")
            # Return only the updates
            return {"current_section": section}
    
    print("DEBUG: All sections complete")
    # Return only the updates
    return {"current_section": "complete"}

def section_research_node(state: AgentState) -> Dict:
    """Conducts research for the current section."""
    section = state["current_section"]
    section_content = state["sections"][section]
    
    messages = [
        SystemMessage(content=SECTION_RESEARCH_PROMPT.format(
            section=section,
            company=state["company_name"],
            content=section_content.content,
            critique=section_content.critique
        )),
        HumanMessage(content=f"Research {section} for {state['company_name']}")
    ]
    
    try:
        response = model_tool.with_structured_output(SearchQueries).invoke(messages)
        if isinstance(response, SearchQueries):
            queries_list = response.queries
        else:
            queries_list = ["company information", "recent news", "financial data"]
            print("Warning: Search queries not properly structured")
        
        new_references = []
        for q in queries_list:
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    response = brave.run(str(q))
                    for r in ast.literal_eval(response):
                        cleaned_snippet = ''.join(c for c in r['snippet'] if c in KEYBOARD_CHARS)
                        ref = Reference(
                            source=r['title'],
                            url=r['link'],
                            snippet=cleaned_snippet,
                            date=datetime.datetime.now().strftime("%Y-%m-%d")
                        )
                        new_references.append(ref)
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Error during search (attempt {retry_count}): {e}")
                        print(f"Retrying in 1 second...")
                        sleep(1)
                    else:
                        print(f"Failed after {max_retries} attempts: {e}")
            sleep(1)
        
        section_content.references.extend(new_references)
        section_content.status = "writing"
        
        # Return only the updates
        return {"sections": state["sections"]}
    except Exception as e:
        print(f"Research error in main try block: {e}")
        return {}  # Return empty updates on error

def section_writing_node(state: AgentState) -> Dict:
    """Writes content for the current section based on research."""
    section = state["current_section"]
    section_content = state["sections"][section]
    
    research_data = "\n".join([
        f"Source: {ref.source}\nURL: {ref.url}\nDate: {ref.date}\nContent: {ref.snippet}\n"
        for ref in section_content.references
    ])
    
    messages = [
        SystemMessage(content=SECTION_WRITING_PROMPT.format(
            section=section,
            research_data=research_data,
            content=section_content.content,
            critique=section_content.critique
        )),
        HumanMessage(content=f"Write {section} section")
    ]
    
    response = model_long.invoke(messages)
    if isinstance(response.content, str):
        section_content.content = response.content
    else:
        section_content.content = str(response.content)
    section_content.status = "reviewing"
    
    # Return only the updates
    return {"sections": state["sections"]}

def section_critique_node(state: AgentState) -> Dict:
    """Reviews and critiques the current section."""
    section = state["current_section"]
    section_content = state["sections"][section]
    
    messages = [
        SystemMessage(content=CRITIQUE_PROMPT.format(
            section=section,
            content=section_content.content
        )),
        HumanMessage(content=f"Review {section} section")
    ]
    
    response = model_tool.invoke(messages)
    if isinstance(response.content, str):
        section_content.critique = response.content
        needs_revision = "needs revision" in response.content.lower()
    else:
        content_str = str(response.content)
        section_content.critique = content_str
        needs_revision = "needs revision" in content_str.lower()
    
    if needs_revision:
        section_content.status = "researching"
    else:
        section_content.status = "complete"
    
    # Return only the updates
    return {"sections": state["sections"]}

def should_continue(state: AgentState) -> str:
    """Determines if processing should continue."""
    if state["current_section"] == "complete":
        return END
    
    section_content = state["sections"][state["current_section"]]
    
    if section_content.status == "pending":
        return "research"
    elif section_content.status == "researching":
        return "research"
    elif section_content.status == "writing":
        return "write"
    elif section_content.status == "reviewing":
        return "critique"
    elif section_content.status == "complete":
        return "plan"
    
    return "plan"

def initialize_state(company_name: str, website: str, max_revisions: int) -> Dict:
    """Initializes the agent state with empty sections."""
    # Initialize sections with proper SectionContent objects
    sections = {}
    for section in REPORT_SECTIONS:
        sections[section] = SectionContent(
            content="",
            status="pending",
            critique="",
            references=[]
        )
    
    # Create and return the state dictionary
    state = {
        "task": f"Research company {company_name} ({website})",
        "company_name": company_name,
        "website": website,
        "sections": sections,
        "current_section": "executive_summary",
        "max_revisions": max_revisions,
        "revision_number": 1
    }
    
    # Debug logging
    print("\nDEBUG: Initial state contents:")
    print(f"Keys in state: {state.keys()}")
    print(f"Sections: {state['sections'].keys()}")
    print(f"Current section: {state['current_section']}")
    
    return state

def validate_state(state: Dict) -> Dict:
    """Validates and fixes state structure if needed."""
    if not isinstance(state, dict):
        print("WARNING: State is not a dictionary, creating new state")
        state = {}
    
    # Ensure all required keys exist
    required_keys = ["task", "company_name", "website", "sections", "current_section", "max_revisions", "revision_number"]
    for key in required_keys:
        if key not in state:
            print(f"WARNING: Missing key {key} in state")
            if key == "sections":
                state[key] = {}
            else:
                state[key] = ""
    
    # Ensure all sections exist and are properly initialized
    if not state["sections"]:
        print("WARNING: Sections dictionary is empty")
        state["sections"] = {}
    
    for section in REPORT_SECTIONS:
        if section not in state["sections"]:
            print(f"WARNING: Missing section {section}")
            state["sections"][section] = SectionContent(
                content="",
                status="pending",
                critique="",
                references=[]
            )
    
    return state

def save_report(state: Dict, company_name: str):
    """Saves the final report to a file."""
    filename = f"{company_name}-final-report.md"
    
    with open(filename, 'w') as f:
        f.write(f"# Market Research Report: {company_name}\n\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n")
        
        for section in REPORT_SECTIONS:
            section_content = state["sections"][section]
            
            # Write section content
            f.write(f"## {section.replace('_', ' ').title()}\n\n")
            f.write(f"{section_content.content}\n\n")
            
            # Write references
            if section_content.references:
                f.write("### References\n\n")
                for ref in section_content.references:
                    f.write(f"- [{ref.source}]({ref.url}) - {ref.date}\n")
                f.write("\n")

def save_state(state: Dict, company_name: str) -> None:
    """Save current state to a JSON file."""
    with open(f"{company_name}_state.json", 'w') as f:
        # Convert complex objects to simple dictionaries
        sections_dict = {}
        for name, section in state["sections"].items():
            if isinstance(section, SectionContent):
                sections_dict[name] = {
                    "content": section.content,
                    "status": section.status,
                    "critique": section.critique,
                    "references": [
                        {
                            "source": ref.source,
                            "url": ref.url,
                            "date": ref.date,
                            "snippet": ref.snippet
                        } for ref in section.references
                    ]
                }
            else:
                print(f"Warning: Section {name} is not a SectionContent object")
                sections_dict[name] = section  # Save as is
        
        # Create serializable state
        save_state = {
            "task": state["task"],
            "company_name": state["company_name"],
            "website": state["website"],
            "current_section": state["current_section"],
            "max_revisions": state["max_revisions"],
            "revision_number": state["revision_number"],
            "sections": sections_dict
        }
        json.dump(save_state, f, indent=2)

def load_state(company_name: str) -> Optional[Dict]:
    """Load state from JSON file if it exists."""
    try:
        with open(f"{company_name}_state.json", 'r') as f:
            saved = json.load(f)
            
            # Reconstruct complex objects
            sections = {}
            for name, section_data in saved["sections"].items():
                if isinstance(section_data, dict):
                    section = SectionContent(
                        content=section_data.get("content", ""),
                        status=section_data.get("status", "pending"),
                        critique=section_data.get("critique", "")
                    )
                    section.references = [
                        Reference(
                            source=ref["source"],
                            url=ref["url"],
                            date=ref["date"],
                            snippet=ref["snippet"]
                        ) for ref in section_data.get("references", [])
                    ]
                    sections[name] = section
                else:
                    print(f"Warning: Loaded section {name} is not in expected format")
                    sections[name] = SectionContent()  # Create empty section
            
            saved["sections"] = sections
            return saved
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading state: {e}")
        return None

def update_state(current_state: Dict, updates: Dict) -> Dict:
    """Helper function to update state while preserving all fields."""
    new_state = current_state.copy()
    for key, value in updates.items():
        if key in new_state:
            new_state[key] = value
    return new_state

def main():
    """Main execution function."""
    # Initialize the workflow graph
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("plan", main_planner_node)
    builder.add_node("research", section_research_node)
    builder.add_node("write", section_writing_node)
    builder.add_node("critique", section_critique_node)
    
    # Set entry point
    builder.set_entry_point("plan")
    
    # Add edges based on the should_continue function
    builder.add_conditional_edges(
        "plan",
        should_continue,
        {
            "research": "research",
            "write": "write",
            "critique": "critique",
            END: END
        }
    )
    
    builder.add_conditional_edges(
        "research",
        should_continue,
        {
            "write": "write",
            "critique": "critique",
            "plan": "plan",
            END: END
        }
    )
    
    builder.add_conditional_edges(
        "write",
        should_continue,
        {
            "critique": "critique",
            "plan": "plan",
            END: END
        }
    )
    
    builder.add_conditional_edges(
        "critique",
        should_continue,
        {
            "research": "research",
            "write": "write",
            "plan": "plan",
            END: END
        }
    )
    
    # Compile the graph
    graph = builder.compile(checkpointer=memory)
    
    # Initialize state
    company_name = "SandboxAQ"  # Replace with actual company name
    website = "https://sandboxaq.com"  # Replace with actual website
    max_revisions = 3
    
    # Create initial state with all sections
    initial_state = {
        "task": f"Research company {company_name} ({website})",
        "company_name": company_name,
        "website": website,
        "sections": {},
        "current_section": "executive_summary",
        "max_revisions": max_revisions,
        "revision_number": 1
    }
    
    # Initialize all sections
    for section in REPORT_SECTIONS:
        initial_state["sections"][section] = SectionContent(
            content="",
            status="pending",
            critique="",
            references=[]
        )
    
    # Create a unique thread ID using timestamp
    thread_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run the workflow
    current_state = initial_state.copy()
    final_state = None
    
    try:
        # Run with thread configuration
        config = {
            "configurable": {
                "thread_id": thread_id
            },
            "recursion_limit": 100
        }
        
        for step_output in graph.stream(current_state, config=config):
            # Update current state with step output while preserving all fields
            if isinstance(step_output, dict):
                for key, value in step_output.items():
                    if key in current_state:
                        current_state[key] = value
            
            print_and_save_multilevel_dict(current_state, filename=f"{company_name}.log")
            save_state(current_state, company_name)  # Save state after each step
            final_state = current_state.copy()
            
    except Exception as e:
        print(f"Error during workflow execution: {e}")
        print(f"Current state keys: {current_state.keys() if current_state else 'None'}")
        if final_state:
            save_state(final_state, company_name)  # Save state on error
        return
    
    # Save final report with proper type casting
    if final_state is not None:
        try:
            # Convert to dictionary with proper types
            state_dict = {
                "task": str(final_state.get("task", "")),
                "company_name": str(final_state.get("company_name", "")),
                "website": str(final_state.get("website", "")),
                "sections": final_state.get("sections", {}),
                "current_section": str(final_state.get("current_section", "")),
                "max_revisions": int(final_state.get("max_revisions", 0)),
                "revision_number": int(final_state.get("revision_number", 0))
            }
            save_report(state_dict, company_name)
        except (ValueError, TypeError, KeyError) as e:
            print(f"Error converting final state: {e}")
    else:
        print("Error: No final state available")

if __name__ == "__main__":
    main() 