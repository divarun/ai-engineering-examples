import streamlit as st
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
import os
from io import BytesIO
from dotenv import load_dotenv

from markdown_pdf import MarkdownPdf, Section
import re # Added for name extraction

# --- 1. Define LLM and Extended State ---
load_dotenv()
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")

# Added check for OLLAMA_BASE_URL for completeness
if not OLLAMA_MODEL_NAME:
    # Use st.error instead of raise EnvironmentError for Streamlit context
    st.error("Missing OLLAMA_MODEL_NAME or OLLAMA_BASE_URL in environment variables.")
    llm = None
else:
    try:
        # NOTE: LangChain's ChatOllama will automatically use OLLAMA_BASE_URL
        llm = ChatOllama(model=OLLAMA_MODEL_NAME)
    except Exception as e:
        st.error(f"Could not initialize Ollama LLM. Please ensure Ollama is running and the '{OLLAMA_MODEL_NAME}' model is downloaded. Error: {e}")
        llm = None

# Define State
class ChainState(TypedDict):
    """Represents the state passed between nodes in the workflow."""
    job_description: str
    original_resume: str
    person_name: str # ADDED: To store the extracted name
    match_score: str # Output of the matching step
    adjusted_resume: str # Output of the adjustment step
    cover_letter: str
    generate_cover_letter: bool # Flag to control routing

# --- 2. Define  Nodes  ---

def analyze_job_match(state: ChainState) -> ChainState:
    """
    Step 1: Analyzes the original resume against the job description for a match.
    """
    if not llm: return state

    # Simple name extraction (assuming the name is on the first line or first word group)
    # This is a basic, non-LLM method for a common use case.
    first_line = state['original_resume'].split('\n')[0].strip()
    # Attempt to find common name patterns (e.g., 2 or 3 words at the start)
    match = re.match(r'^([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})', first_line)
    name = match.group(0).strip() if match else "Candidate Name"

    prompt = f"""
You are an expert career consultant. Analyze the following resume against the job description. 
Provide a detailed analysis (3-5 sentences) of the strongest matches and the most critical gaps. 
Finally, give a confidence score (0-100) and state whether the candidate is a 'Strong Match', 'Moderate Match', or 'Poor Match'.

Job Description:
{state['job_description']}

Resume:
{state['original_resume']}
"""
    with st.spinner("🔍 Step 1: Analyzing Job Match..."):
        response = llm.invoke(prompt)

    st.markdown("---")
    st.info("### 🔍 Match Analysis Complete!")
    st.markdown(response.content.strip())

    # ADDED: Store the extracted name
    return {**state, "match_score": response.content.strip(), "person_name": name}


def adjust_resume(state: ChainState) -> ChainState:
    """
    Step 2: Adjusts the resume to be highly ATS-friendly, prioritizing simple formatting
    and keyword matching from the job description.
    """
    if not llm: return state

    # Get the extracted name
    name_header = state.get('person_name', 'Candidate Name')

    prompt = f"""
You are an expert **ATS (Applicant Tracking System) Specialist** and resume tailor. 
Your primary goal is to maximize the resume's match score against the job description for a machine reader.

**Instructions for ATS Optimization:**
1.  **Header:** The very first line of the output MUST be the candidate's name: **{name_header}**.
2.  **Formatting:** Use only plain text, simple bullet points, and standard section headers (e.g., 'EXPERIENCE', 'SKILLS', 'EDUCATION'). Avoid tables, graphics, or fancy fonts.
3.  **Keywords:** Directly incorporate specific keywords and phrases from the **Job Description** into the resume's Summary, Skills, and Experience sections.
4.  **Clarity:** Ensure every bullet point in the experience section starts with a strong action verb and quantifiable achievements are highlighted.
5.  **Content:** Based on the Job Match analysis, revise the provided original resume to better highlight relevant experience that aligns with the job description. Do not invent new experience, only rephrase existing content to be more impactful and relevant to the target job.

Job Description:
{state['job_description']}

Job Match Analysis:
{state['match_score']}

Original Resume:
{state['original_resume']}

**OUTPUT THE REVISED, ATS-FRIENDLY RESUME TEXT ONLY. STARTING WITH THE NAME.**
"""
    with st.spinner("✍️ Step 2: Adjusting Resume for Target Job & ATS Compliance..."):
        response = llm.invoke(prompt)

    return {**state, "adjusted_resume": response.content.strip()}


def generate_cover_letter(state: ChainState) -> ChainState:
    """
    Step 3 (Optional): Generates a cover letter using the adjusted resume and job description.
    """
    if not llm: return state

    prompt = f"""
You're a professional cover letter writing assistant. Write a professional and personalized cover letter for the job, using the tailored resume content below to structure the key arguments.

Job Description:
{state['job_description']}

Tailored Resume Content:
{state['adjusted_resume']}
"""
    with st.spinner("✉️ Step 3: Generating Cover Letter..."):
        response = llm.invoke(prompt)

    return {**state, "cover_letter": response.content.strip()}

# --- 3. Conditional Routing Function  ---

def route_to_cover_letter(state: ChainState) -> str:
    """Decides whether to proceed to cover letter generation or finish."""
    if state.get("generate_cover_letter"):
        return "generate_cover_letter"
    else:
        return END

# --- 4. LangGraph Workflow Setup  ---

@st.cache_resource
def compile_workflow():
    workflow = StateGraph(ChainState)

    # Add Nodes
    workflow.add_node("analyze_job_match", analyze_job_match)
    workflow.add_node("adjust_resume", adjust_resume)
    workflow.add_node("generate_cover_letter", generate_cover_letter)

    # Define Edges/Flow
    workflow.set_entry_point("analyze_job_match")
    workflow.add_edge("analyze_job_match", "adjust_resume")

    # Conditional Edge (Router)
    workflow.add_conditional_edges(
        "adjust_resume", # Source node
        route_to_cover_letter, # Router function
        {
            "generate_cover_letter": "generate_cover_letter", # Path if true
            END: END # Path if false
        }
    )
    return workflow.compile()

app = compile_workflow()

# --- 5. Streamlit UI  ---

# ---  HELPER FUNCTION TO GENERATE PDF ---

def generate_pdf(markdown_content: str, title: str) -> BytesIO:
    """Converts markdown content into a PDF file in memory."""
    pdf = MarkdownPdf()
    # Add a section. Setting 'toc=False' prevents a table of contents from being generated for a simple document.
    pdf.add_section(Section(markdown_content, toc=False))

    # Optional: Set PDF metadata
    pdf.meta["title"] = title

    # Save to an in-memory byte buffer
    buffer = BytesIO()
    pdf.save_bytes(buffer)
    buffer.seek(0) # Rewind the buffer to the beginning
    return buffer

# --- Main App Logic ---

st.title("💼 AI Job Application & Match Consultant")
st.markdown("Analyze job fit and tailor your resume using a multi-step LangGraph workflow.")
st.warning(f"⚠️ **Requirement:** This app requires **Ollama** to be running locally with the `{OLLAMA_MODEL_NAME}` model downloaded.")

# --- Input Area ---
col1, col2 = st.columns(2)

with col1:
    job_description_input = st.text_area(
        "📝 **Job Description**",
        value="We are seeking a Senior Python Developer with 5+ years of experience in API development (Flask/Django), cloud deployment (AWS), and microservices architecture. Experience with CI/CD is a plus.",
        height=300
    )

with col2:
    resume_input = st.text_area(
        "📄 **Your Original Resume** (Paste full text content)",
        value="John Doe - Python Developer\n\nContact: john.doe@email.com\n\nSkills: Python, Flask, basic Docker. I have built a few web apps and enjoy coding. Not much cloud experience but a fast learner.",
        height=300
    )

# Options and Trigger
st.markdown("---")
generate_cl = st.checkbox("✉️ **Generate Cover Letter** (Optional Step 3)", value=True)

if st.button("🚀 Run Analysis & Tailoring Workflow", type="primary"):
    if not llm:
        st.error("The LLM is not initialized. Please ensure Ollama is running.")
    elif job_description_input.strip() == "" or resume_input.strip() == "":
        st.error("Please enter both the job description and your resume content to proceed.")
    else:
        # Run the LangGraph workflow
        try:
            initial_state = {
                "job_description": job_description_input.strip(),
                "original_resume": resume_input.strip(),
                "generate_cover_letter": generate_cl, # Pass the boolean flag
                "match_score": "", "adjusted_resume": "", "cover_letter": "", "person_name": "" # Initialize
            }

            # The workflow execution
            st.info("--- Starting Workflow: Step 1/2 (or 3) ---")
            result = app.invoke(initial_state)

            # --- Display Results ---
            st.success("✅ Workflow Complete! Review your tailored materials below.")

            # --- RESUME OUTPUT & DOWNLOAD OPTIONS ---
            st.header("✨ Step 2: Tailored ATS Resume")
            adjusted_resume_content = result['adjusted_resume']

            # Use st.text_area for easy copy/paste
            st.text_area("ATS Resume Content (Copy/Paste)", adjusted_resume_content, height=400)

            st.subheader("⬇️ Download & Conversion Options")

            # 1. PDF Option (Direct Download)
            resume_pdf_data = generate_pdf(adjusted_resume_content, f"ATS Resume - {result.get('person_name', 'Document')}")
            st.download_button(
                label="📥 **Option A: Download as PDF**",
                data=resume_pdf_data,
                file_name="tailored_ats_resume.pdf",
                mime="application/pdf"
            )

            # 2. Word/Google Docs Option (Instructions + Copy button)
            st.markdown(
                """
                **✍️ Option B: Copy to Word / Google Docs**
                1. Click the **"Copy Text"** button below.
                2. Open **Microsoft Word** or a **Google Doc**.
                3. Paste the text. *You may need to manually clean up minor formatting.*
                """
            )
            # Use a button to copy text to clipboard (requires a Streamlit feature or manual instruction)
            st.button("📋 **Copy Text** (Use the text area above for copy/paste)", disabled=True) # Text area is better for copy

            # --- COVER LETTER OUTPUT & DOWNLOAD OPTIONS (Conditional) ---
            if generate_cl:
                st.markdown("---")
                st.header("✉️ Step 3: Generated Cover Letter")
                cover_letter_content = result['cover_letter']

                # Use st.text_area for easy copy/paste
                st.text_area("Cover Letter Content (Copy/Paste)", cover_letter_content, height=400)

                st.subheader("⬇️ Download & Conversion Options (Cover Letter)")

                # 1. PDF Option (Direct Download)
                cl_pdf_data = generate_pdf(cover_letter_content, f"Cover Letter - {result.get('person_name', 'Document')}")
                st.download_button(
                    label="📥 **Option A: Download as PDF**",
                    data=cl_pdf_data,
                    file_name="tailored_cover_letter.pdf",
                    mime="application/pdf"
                )

                # 2. Word/Google Docs Option (Instructions + Copy button)
                st.markdown(
                    """
                    **✍️ Option B: Copy to Word / Google Docs**
                    1. Copy the text from the text area above.
                    2. Paste it into **Microsoft Word** or a **Google Doc**.
                    """
                )

        except Exception as e:
            st.error(f"An error occurred during workflow execution: {e}")