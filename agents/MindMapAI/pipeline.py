import os
from datetime import datetime
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool
from config import get_llm

def generate_content(url: str, agent_timeout_sec: int, agent_max_iter: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"mermaid_code_{timestamp}.txt"

    llm = get_llm()
    scrape_tool = ScrapeWebsiteTool()

    researcher_agent = Agent(
        role="Web Content Scraper",
        goal="Extract clean, full-text main article content only.",
        backstory="Expert at using web scraping tools to extract content from URLs and clean the text for analysis.",
        tools=[scrape_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    analysis_agent = Agent(
        role="Content Analysis Specialist",
        goal="Extract a structured mind map hierarchy from article text.",
        backstory="Expert at analyzing text content and breaking down articles into structured mind map hierarchies.",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    mapper_agent = Agent(
        role="Mermaid Code Generator",
        goal="Generate ONLY valid Mermaid.js code starting with 'graph TD' - no text, no explanations, no thoughts.",
        backstory="Expert at converting structured data into precise Mermaid.js diagram code. "
                  "Outputs ONLY the raw Mermaid code starting with 'graph TD'. Never includes any text before or after the code. "
                  "Never includes thoughts, explanations, reasoning, or any other text - just the pure Mermaid diagram code.",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    scraping_task = Task(
        description=f"""
        Scrape ONLY the main article body from: {url}
        - Ignore navigation, headers, footers, sidebars, cookie banners, newsletter prompts, comments, ads, and related links.
        - Keep paragraph breaks.
        Use the 'Read website content' tool once with: {{"website_url": "{url}"}}
        """,
        expected_output="Clean full-text of the article.",
        agent=researcher_agent,
        tools=[scrape_tool],
        max_iterations=agent_max_iter,
        max_execution_time=agent_timeout_sec
    )

    analysis_task = Task(
        description=(
            "From the scraped main-article text, extract a mind map hierarchy:\n"
            "- 1 Main Topic\n- 3–5 Subtopics\n- 2–3 Key Ideas per Subtopic\n"
            "Rules:\n"
            "- Use indented bullet list format.\n"
            "- Each label under 7 words, based on the article.\n"
            "- Discard unrelated or uncertain items.\n"
            "Limits:\n"
            "- Keep total nodes (1 main + subtopics + ideas) ≤ 20.\n"
            "- Keep this bullet list concise (≤ 200 words)."
        ),
        expected_output="Indented bullet list of mind map topics.",
        agent=analysis_agent,
        context=[scraping_task],
        max_iterations=agent_max_iter,
        max_execution_time=agent_timeout_sec
    )

    validation_task = Task(
        description=(
            "Validate the hierarchy from previous task:\n"
            "- Ensure all Subtopics and Key Ideas are grounded in the original scraped article.\n"
            "- Remove ungrounded items.\n"
            "- Keep the structure and labels concise."
        ),
        expected_output="Validated bullet list grounded in article.",
        agent=analysis_agent,
        context=[scraping_task, analysis_task],
        max_iterations=agent_max_iter,
        max_execution_time=agent_timeout_sec
    )

    mapping_task = Task(
        description=(
            "Convert the validated bullet hierarchy into valid Mermaid.js mind map code.\n\n"
            "CRITICAL: Your response must start with 'graph TD' and contain ONLY Mermaid code. "
            "Do NOT include any text before or after the code. Do NOT include thoughts, explanations, "
            "or reasoning. Do NOT include phrases like 'Here is the final answer' or 'I will not provide a thought'.\n\n"
            "FORMAT REQUIREMENTS:\n"
            "1. Start with 'graph TD'\n"
            "2. Main topic: A((Topic Name))\n"
            "3. Subtopics: A --> B1(Subtopic Name)\n"
            "4. Key ideas: B1 --> C1(Key Idea Name)\n"
            "5. Each node must have a valid ID followed by its label in parentheses — "
            "for example, 'B1(Subtopic Name)', NOT '(B1SubtopicName)'.\n"
            "6. Do NOT include the node ID inside the parentheses or brackets.\n"
            "7. Use parentheses ( ... ) for normal nodes and double parentheses (( ... )) only for the main topic.\n"
            "8. Keep total nodes <= 20.\n\n"
            "SANITIZATION RULES (to prevent Mermaid parse errors):\n"
            "9. Replace any double quotes (\") with single quotes (').\n"
            "10. Never use parentheses inside node labels. Inner parentheses like (GNNs) will cause Mermaid to break. "
            "Replace them with commas, or remove them completely. "
            "For example, use 'Graph Neural Networks , GNNs' or 'Graph Neural Networks GNNs'. "
            "Do NOT output: 'Graph Neural Networks (GNNs)'.\n"
            "11. Replace special symbols like $, :, and - with spaces.\n"
            "12. Convert all smart quotes (' ' \" \") to straight quotes (' ').\n"
            "13. Remove or replace any non-ASCII characters (like curly dashes or typographic apostrophes) with standard ones.\n\n"
            "14. Ensure that each label ends with exactly one closing parenthesis (')' or '))' depending on the node type. "
            "Do NOT include extra closing parentheses inside the label or at the end. Mermaid cannot parse labels like "
            "'Graph Neural Networks (GNNs))'.\n\n"
            "OUTPUT: ONLY the raw Mermaid code starting with 'graph TD' — nothing else.\n\n"
            "EXAMPLE:\n"
            "graph TD\n"
            "A((AI Career Coaching))\n"
            "A --> B1(Prompt Engineering)\n"
            "A --> B2(Goal Setting)\n"
            "B1 --> C1(Specific Prompts)\n"
            "B1 --> C2(Context Matters)"
        ),
        expected_output="Complete Mermaid.js code starting with 'graph TD' and containing the mind map structure",
        agent=mapper_agent,
        context=[validation_task],
        max_iterations=agent_max_iter,
        max_execution_time=agent_timeout_sec
    )


    crew = Crew(
        agents=[researcher_agent, analysis_agent, mapper_agent],
        tasks=[scraping_task, analysis_task, validation_task, mapping_task],
        verbose=True
    )

    try:
        crew.kickoff()
    except Exception as e:
        return f"ERROR: CrewAI failed — {str(e)}"

    # Get the mapping task output
    mermaid_output = mapping_task.output
    mermaid_code = getattr(mermaid_output, "raw", str(mermaid_output))
    
    if isinstance(mermaid_code, str) and mermaid_code.strip():
        mermaid_code = mermaid_code.strip()
        
        # Clean up any markdown formatting
        if "```mermaid" in mermaid_code:
            start = mermaid_code.find("```mermaid") + 10
            end = mermaid_code.find("```", start)
            if end != -1:
                mermaid_code = mermaid_code[start:end].strip()
        elif "```" in mermaid_code:
            mermaid_code = mermaid_code.replace("```", "").strip()
        
        # Find the first occurrence of "graph TD" and extract from there
        graph_start = mermaid_code.find("graph TD")
        if graph_start != -1:
            mermaid_code = mermaid_code[graph_start:].strip()
        elif not mermaid_code.startswith("graph"):
            mermaid_code = "graph TD\n" + mermaid_code
        
        # Validate that we have actual Mermaid content
        if "graph" in mermaid_code and "-->" in mermaid_code:
            with open(output_file, "w") as f:
                f.write(mermaid_code)
            return mermaid_code
        else:
            return f"ERROR: Generated content doesn't appear to be valid Mermaid code. Got: {mermaid_code[:200]}..."
    else:
        return f"ERROR: Mermaid code not generated or is empty. Output was: {mermaid_output}"

