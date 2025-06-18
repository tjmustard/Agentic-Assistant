from dotenv import load_dotenv
_ = load_dotenv()

import os
import io
import json
import datetime
import sqlite3
import ast
from time import sleep
from typing import TypedDict, List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pydantic import BaseModel
import argparse

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
    human_feedback: str = ""
    focus_areas: List[str] = field(default_factory=list)
    iteration_count: int = 0  # Track iterations for this section

class AgentState(TypedDict):
    """Represents the state of the agent system"""
    task: str  # Company research task
    company_name: str
    website: str
    sections: Dict[str, SectionContent]
    current_section: str
    max_revisions: int
    max_iterations: int
    revision_number: int
    human_instructions: str  # New field for overall human instructions
    custom_sections: List[str]  # New field for human-added sections
    cycle_number: int  # Track which cycle we're on
    cycle_instructions: Dict[int, str]  # Instructions for each cycle

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
4. Consider any human feedback and instructions

The available sections are:
{sections}

Consider dependencies (e.g., Executive Summary should be last) and information flow between sections.
Pay special attention to any human instructions or focus areas provided."""

SECTION_RESEARCH_PROMPT = """You are a specialized research agent focused on gathering information for the {section} 
section of a market research report about {company}. Your task is to:
1. Generate specific search queries to gather relevant information
2. Focus on finding verifiable facts and data
3. Include source URLs and dates for all information
4. Ensure information is recent and relevant
5. Look for specific metrics and quantifiable data
6. Find competitor information where relevant

Human feedback and focus areas: {human_feedback}
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
7. Address any specific human feedback or focus areas

Human feedback and focus areas: {human_feedback}
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
7. Alignment with human feedback and focus areas

Human feedback: {human_feedback}
Section: {section}
Content: {content}"""

# Add verbosity control class
class VerbosityConfig:
    """Controls logging verbosity throughout the application."""
    def __init__(self, verbose: bool = False, debug: bool = False):
        self.verbose = verbose  # Detailed information
        self.debug = debug    # Debug-level information
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message based on verbosity settings."""
        if level == "debug" and not self.debug:
            return
        if level == "verbose" and not self.verbose:
            return
        print(message)
    
    def log_state(self, state: Dict[str, Any], level: str = "info") -> None:
        """Log state information based on verbosity settings."""
        if level == "debug" and not self.debug:
            return
        if level == "verbose" and not self.verbose:
            return
        
        current_section = state.get("current_section", "unknown")
        print("\nCurrent State:")
        print(f"Section: {current_section}")
        
        if current_section in state.get("sections", {}):
            section_content = state["sections"][current_section]
            print(f"Status: {section_content.status}")
            print(f"Iteration: {section_content.iteration_count}")
            
            if self.verbose:
                print("\nAll Sections Status:")
                for section, content in state["sections"].items():
                    print(f"- {section}: {content.status} (iterations: {content.iteration_count})")
        
        if self.debug:
            print("\nFull State Details:")
            print_and_save_multilevel_dict(state)

# Create global config
config = VerbosityConfig()

def print_and_save_multilevel_dict(dictionary, filename=None, indent=0):
    """Prints and saves the contents of a multi-level dictionary."""
    if filename and os.path.exists(filename):
        append_write = 'a'
    else:
        append_write = 'w'

    def _print_and_save_multilevel_dict(dictionary, indent, file=None):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                line = f"{' ' * indent}{key}:"
                if config.debug:
                    print(line)
                if file:
                    file.write(line + "\n")
                _print_and_save_multilevel_dict(value, indent + 4, file)
            else:
                line = f"{' ' * indent}{key} = {value}"
                if config.debug:
                    print(line)
                if file:
                    file.write(line + "\n")

    if filename:
        with open(filename, append_write) as f:
            _print_and_save_multilevel_dict(dictionary, indent, f)
    else:
        _print_and_save_multilevel_dict(dictionary, indent)

def get_human_feedback() -> Tuple[str, List[str], List[str]]:
    """Get feedback and instructions from the human user."""
    print("\n=== Human Feedback Required ===")
    print("Please provide your feedback and instructions:")
    print("1. Enter any general feedback/instructions (press Enter twice to finish)")
    print("2. Enter any specific focus areas (one per line, press Enter twice to finish)")
    print("3. Enter any new sections to add (one per line, press Enter twice to finish)")
    
    # Get general feedback
    print("\nGeneral feedback/instructions:")
    feedback_lines = []
    while True:
        line = input()
        if line == "":
            break
        feedback_lines.append(line)
    feedback = "\n".join(feedback_lines)
    
    # Get focus areas
    print("\nSpecific focus areas (press Enter twice to finish):")
    focus_areas = []
    while True:
        line = input()
        if line == "":
            break
        focus_areas.append(line)
    
    # Get new sections
    print("\nNew sections to add (press Enter twice to finish):")
    new_sections = []
    while True:
        line = input()
        if line == "":
            break
        new_sections.append(line.lower().replace(" ", "_"))
    
    return feedback, focus_areas, new_sections

def print_section_summary(state: AgentState, section: str) -> None:
    """Print a summary of the current section for human review."""
    section_content = state["sections"][section]
    print(f"\n=== Section Summary: {section} ===")
    print(f"Status: {section_content.status}")
    
    if config.verbose:
        print(f"Content length: {len(section_content.content)} characters")
        print(f"Number of references: {len(section_content.references)}")
        print("\nContent preview (first 500 characters):")
        print(section_content.content[:500] + "..." if len(section_content.content) > 500 else section_content.content)
        print("\nCurrent focus areas:", section_content.focus_areas)
        print("\nPrevious human feedback:", section_content.human_feedback)

def update_state(current_state: Dict, updates: Dict) -> Dict:
    """Helper function to update state while preserving all fields."""
    new_state = current_state.copy()
    for key, value in updates.items():
        if key in new_state:
            new_state[key] = value
    return new_state

def main_planner_node(state: AgentState) -> Dict:
    """Determines the next section to work on and updates the state."""
    config.log("\nPlanning next section...", "info")
    config.log(f"State type: {type(state)}", "debug")
    config.log(f"State keys: {state.keys() if isinstance(state, dict) else 'Not a dict'}", "debug")
    
    messages = [
        SystemMessage(content=MAIN_PLANNER_PROMPT.format(sections="\n".join(REPORT_SECTIONS))),
        HumanMessage(content=f"Current state: {state}")
    ]
    response = model_tool.invoke(messages)
    
    # Initialize any missing sections
    for section in REPORT_SECTIONS + state.get("custom_sections", []):
        if section not in state["sections"]:
            config.log(f"Adding missing section {section}", "debug")
            state["sections"][section] = SectionContent(
                content="",
                status="pending",
                critique="",
                references=[],
                human_feedback=state["cycle_instructions"].get(state["cycle_number"], ""),
                focus_areas=[]
            )
        
        if state["sections"][section].status == "pending":
            config.log(f"Selected section: {section}", "info")
            # Create a new state copy with updated current_section
            new_state = state.copy()
            new_state["current_section"] = section
            return new_state
    
    config.log("All sections complete", "info")
    new_state = state.copy()
    new_state["current_section"] = "complete"
    return new_state

def section_research_node(state: AgentState) -> Dict:
    """Conducts research for the current section."""
    section = state["current_section"]
    section_content = state["sections"][section]
    
    messages = [
        SystemMessage(content=SECTION_RESEARCH_PROMPT.format(
            section=section,
            company=state["company_name"],
            content=section_content.content,
            critique=section_content.critique,
            human_feedback=section_content.human_feedback
        )),
        HumanMessage(content=f"Research {section} for {state['company_name']}")
    ]
    
    try:
        response = model_tool.with_structured_output(SearchQueries).invoke(messages)
        if isinstance(response, SearchQueries):
            queries_list = response.queries
        else:
            queries_list = ["company information", "recent news", "financial data"]
            config.log("Warning: Search queries not properly structured", "debug")
        
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
                        config.log(f"Error during search (attempt {retry_count}): {e}", "debug")
                        config.log("Retrying in 1 second...", "debug")
                        sleep(1)
                    else:
                        config.log(f"Failed after {max_retries} attempts: {e}", "debug")
            sleep(1)
        
        section_content.references.extend(new_references)
        section_content.status = "writing"
        
        return {"sections": state["sections"]}
    except Exception as e:
        config.log(f"Research error in main try block: {e}", "debug")
        return {}

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
            critique=section_content.critique,
            human_feedback=section_content.human_feedback
        )),
        HumanMessage(content=f"Write {section} section")
    ]
    
    response = model_long.invoke(messages)
    if isinstance(response.content, str):
        section_content.content = response.content
    else:
        section_content.content = str(response.content)
    section_content.status = "reviewing"
    
    return {"sections": state["sections"]}

def section_critique_node(state: AgentState) -> Dict:
    """Reviews and critiques the current section."""
    section = state["current_section"]
    section_content = state["sections"][section]
    
    messages = [
        SystemMessage(content=CRITIQUE_PROMPT.format(
            section=section,
            content=section_content.content,
            human_feedback=section_content.human_feedback
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
    
    # Check if we should revise based on iteration count
    if needs_revision and section_content.iteration_count < state["max_iterations"]:
        section_content.status = "researching"
        section_content.iteration_count += 1
        config.log(f"Starting iteration {section_content.iteration_count} for section {section}", "info")
    else:
        if needs_revision:
            config.log(f"Section {section} needs revision but reached max iterations ({state['max_iterations']})", "info")
        section_content.status = "complete"
        config.log(f"Section {section} completed after {section_content.iteration_count} iterations", "info")
    
    return {"sections": state["sections"]}

def get_cycle_feedback(state: AgentState) -> Tuple[bool, str, List[str]]:
    """Get feedback after completing all sections and decide whether to continue."""
    print("\n=== Cycle Complete ===")
    print(f"Completed cycle {state['cycle_number']}")
    
    # Show summary of all sections
    print("\nSection Status:")
    for section, content in state["sections"].items():
        print(f"\n{section.replace('_', ' ').title()}:")
        print(f"- Status: {content.status}")
        print(f"- Iterations: {content.iteration_count}")
        print(f"- References: {len(content.references)}")
        if config.verbose:
            print("- Content Preview:")
            preview = content.content[:200] + "..." if len(content.content) > 200 else content.content
            print(f"  {preview}")
    
    print("\nWould you like to run another cycle with additional instructions? (yes/no)")
    continue_cycles = input().lower().startswith('y')
    
    if continue_cycles:
        print("\nPlease provide instructions for the next cycle.")
        print("These instructions will be used by all agents for all sections.")
        print("Enter your instructions (press Enter twice to finish):")
        instruction_lines = []
        while True:
            line = input()
            if line == "":
                break
            instruction_lines.append(line)
        instructions = "\n".join(instruction_lines)
        
        print("\nWould you like to add any new sections? (Enter section names, one per line, press Enter twice to finish)")
        new_sections = []
        while True:
            line = input()
            if line == "":
                break
            if line:
                new_sections.append(line.lower().replace(" ", "_"))
        
        return continue_cycles, instructions, new_sections
    
    return continue_cycles, "", []

def print_cycle_summary(state: AgentState) -> None:
    """Print a summary of the current cycle."""
    print(f"\n=== Cycle {state['cycle_number']} Summary ===")
    print(f"Instructions for this cycle: {state['cycle_instructions'].get(state['cycle_number'], 'None')}")
    
    if config.verbose:
        print("\nSection Status:")
        for section, content in state["sections"].items():
            print(f"- {section}: {content.status}")
            if content.status == "complete":
                print(f"  Length: {len(content.content)} chars")
                print(f"  Iterations: {content.iteration_count}")
                print(f"  References: {len(content.references)}")

def initialize_state(company_name: str, website: str, max_revisions: int, max_iterations: int) -> Dict:
    """Initializes the agent state with empty sections."""
    print("\n=== Initial Setup ===")
    print("Before we begin, let's configure the research scope.")
    print("Please provide initial instructions and any custom sections you'd like to add.")
    feedback, focus_areas, custom_sections = get_human_feedback()
    
    # Initialize sections with proper SectionContent objects
    sections = {}
    all_sections = REPORT_SECTIONS + custom_sections
    
    for section in all_sections:
        sections[section] = SectionContent(
            content="",
            status="pending",
            critique="",
            references=[],
            human_feedback=feedback,
            focus_areas=focus_areas.copy(),
            iteration_count=0
        )
    
    return {
        "task": f"Research company {company_name} ({website})",
        "company_name": company_name,
        "website": website,
        "sections": sections,
        "current_section": "executive_summary",
        "max_revisions": max_revisions,
        "max_iterations": max_iterations,
        "revision_number": 1,
        "human_instructions": feedback,
        "custom_sections": custom_sections,
        "cycle_number": 1,
        "cycle_instructions": {1: feedback}
    }

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
                    "human_feedback": section.human_feedback,
                    "focus_areas": section.focus_areas,
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
                sections_dict[name] = section
        
        # Create serializable state
        save_state = {
            "task": state["task"],
            "company_name": state["company_name"],
            "website": state["website"],
            "current_section": state["current_section"],
            "max_revisions": state["max_revisions"],
            "max_iterations": state["max_iterations"],
            "revision_number": state["revision_number"],
            "human_instructions": state.get("human_instructions", ""),
            "custom_sections": state.get("custom_sections", []),
            "cycle_number": state.get("cycle_number", 1),
            "cycle_instructions": state.get("cycle_instructions", {}),
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
                        critique=section_data.get("critique", ""),
                        human_feedback=section_data.get("human_feedback", ""),
                        focus_areas=section_data.get("focus_areas", []),
                        iteration_count=section_data.get("iteration_count", 0)
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
                    sections[name] = SectionContent()
            
            saved["sections"] = sections
            return saved
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading state: {e}")
        return None

def save_report(state: Dict, company_name: str):
    """Saves the final report to a file."""
    filename = f"{company_name}-final-report.md"
    
    with open(filename, 'w') as f:
        f.write(f"# Market Research Report: {company_name}\n\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n")
        
        # Write human instructions if available
        if "human_instructions" in state and state["human_instructions"]:
            f.write("## Research Focus and Instructions\n\n")
            f.write(state["human_instructions"] + "\n\n")
        
        # Write each section
        for section in (REPORT_SECTIONS + state.get("custom_sections", [])):
            if section in state["sections"]:
                section_content = state["sections"][section]
                
                # Write section content
                f.write(f"## {section.replace('_', ' ').title()}\n\n")
                if section_content.focus_areas:
                    f.write("### Focus Areas\n")
                    for area in section_content.focus_areas:
                        f.write(f"- {area}\n")
                    f.write("\n")
                
                f.write(f"{section_content.content}\n\n")
                
                # Write references
                if section_content.references:
                    f.write("### References\n\n")
                    for ref in section_content.references:
                        f.write(f"- [{ref.source}]({ref.url}) - {ref.date}\n")
                    f.write("\n")

def should_continue(state: AgentState) -> str:
    """Determines if processing should continue."""
    if state["current_section"] == "complete":
        # All sections are done for this cycle
        config.log(f"\nCompleted cycle {state['cycle_number']}", "info")
        
        # Show summary of all sections
        print("\nCycle Summary:")
        for section, content in state["sections"].items():
            print(f"\n{section.replace('_', ' ').title()}:")
            print(f"- Status: {content.status}")
            print(f"- Iterations: {content.iteration_count}")
            print(f"- References: {len(content.references)}")
            if config.verbose:
                print("- Content Preview:")
                preview = content.content[:200] + "..." if len(content.content) > 200 else content.content
                print(f"  {preview}")
        
        # Get feedback for next cycle
        print("\nWould you like to run another cycle with additional instructions? (yes/no)")
        continue_cycles = input().lower().startswith('y')
        
        if not continue_cycles:
            return END
        
        # Get instructions for next cycle
        print("\nPlease provide instructions for the next cycle.")
        print("These instructions will be used by all agents for all sections.")
        print("Enter your instructions (press Enter twice to finish):")
        instructions = []
        while True:
            line = input()
            if line == "":
                break
            instructions.append(line)
        cycle_instructions = "\n".join(instructions)
        
        # Update state for next cycle
        state["cycle_number"] += 1
        state["cycle_instructions"][state["cycle_number"]] = cycle_instructions
        
        # Reset all sections to pending and update with new instructions
        for section in state["sections"].values():
            section.status = "pending"
            section.human_feedback = cycle_instructions
            section.iteration_count = 0  # Reset iteration count for new cycle
        
        return "plan"
    
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

def main():
    """Main execution function."""
    # Add command line arguments
    parser = argparse.ArgumentParser(description='Market Research Agent')
    parser.add_argument('--company', type=str, default="SandboxAQ", help='Company name')
    parser.add_argument('--website', type=str, default="https://sandboxaq.com", help='Company website')
    parser.add_argument('--max-revisions', type=int, default=3, help='Maximum revisions per section')
    parser.add_argument('--max-iterations', type=int, default=3, help='Maximum iterations per section before moving on')
    parser.add_argument('--max-cycles', type=int, default=3, help='Maximum number of cycles')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Print configuration header
    print("\n" + "="*50)
    print(" Market Research Agent Configuration")
    print("="*50)
    print(f"{'Parameter':<20} {'Value':<30}")
    print("-"*50)
    print(f"{'Company':<20} {args.company:<30}")
    print(f"{'Website':<20} {args.website:<30}")
    print(f"{'Max Revisions':<20} {args.max_revisions:<30}")
    print(f"{'Max Iterations':<20} {args.max_iterations:<30}")
    print(f"{'Max Cycles':<20} {args.max_cycles:<30}")
    print(f"{'Verbose Mode':<20} {'Enabled' if args.verbose else 'Disabled':<30}")
    print(f"{'Debug Mode':<20} {'Enabled' if args.debug else 'Disabled':<30}")
    print("="*50 + "\n")
    
    # Configure verbosity
    global config
    config = VerbosityConfig(verbose=args.verbose, debug=args.debug)
    
    config.log(f"Starting research for {args.company}", "info")
    config.log(f"Full configuration: {args}", "debug")
    
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
    company_name = args.company
    website = args.website
    max_revisions = args.max_revisions
    max_iterations = args.max_iterations
    
    # Try to load existing state or create new one
    try:
        saved_state = load_state(company_name)
        if saved_state:
            print(f"\nFound existing research in progress. Would you like to:")
            print("1. Continue from where you left off")
            print("2. Start fresh")
            choice = input("Enter 1 or 2: ")
            
            if choice == "1":
                print(f"Resuming from saved state. Current section: {saved_state.get('current_section', 'unknown')}")
                # Get any new instructions
                feedback, focus_areas, new_sections = get_human_feedback()
                saved_state["human_instructions"] = feedback
                saved_state["custom_sections"].extend(new_sections)
                saved_state["max_iterations"] = max_iterations  # Add max_iterations to saved state
                initial_state = saved_state
            else:
                print("Starting new research process")
                initial_state = initialize_state(company_name, website, max_revisions, max_iterations)
        else:
            print("Starting new research process")
            initial_state = initialize_state(company_name, website, max_revisions, max_iterations)
    except Exception as e:
        print(f"Error loading state: {e}")
        print("Starting new research process")
        initial_state = initialize_state(company_name, website, max_revisions, max_iterations)
    
    # Create a unique thread ID using timestamp
    thread_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run the workflow
    current_state = initial_state.copy()
    final_state = None
    
    try:
        # Run with thread configuration
        config_dict: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id
            },
            "recursion_limit": 100
        }
        
        current_state = initial_state.copy()
        
        for step_output in graph.stream(current_state, config=config_dict):
            # Update current state with step output
            if isinstance(step_output, dict):
                # Create a new state copy with updates
                new_state = current_state.copy()
                for key, value in step_output.items():
                    if key in new_state:
                        new_state[key] = value
                current_state = new_state
            
            # Check if we've exceeded max cycles
            if current_state["cycle_number"] > args.max_cycles:
                config.log(f"Reached maximum number of cycles ({args.max_cycles})", "info")
                break
            
            config.log_state(current_state)
            if config.debug:
                print_and_save_multilevel_dict(current_state, filename=f"{company_name}.log")
            save_state(current_state, company_name)
            final_state = current_state.copy()
            
    except Exception as e:
        config.log(f"Error during workflow execution: {e}", "info")
        config.log(f"Current state keys: {current_state.keys() if current_state else 'None'}", "debug")
        if final_state:
            save_state(final_state, company_name)
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
                "max_iterations": int(final_state.get("max_iterations", 0)),
                "revision_number": int(final_state.get("revision_number", 0)),
                "human_instructions": str(final_state.get("human_instructions", "")),
                "custom_sections": final_state.get("custom_sections", []),
                "cycle_number": int(final_state.get("cycle_number", 1)),
                "cycle_instructions": final_state.get("cycle_instructions", {})
            }
            save_report(state_dict, company_name)
        except (ValueError, TypeError, KeyError) as e:
            print(f"Error converting final state: {e}")
    else:
        print("Error: No final state available")

if __name__ == "__main__":
    main() 