import traceback
import streamlit as st

from utils import is_valid_url
from pipeline import generate_content
from config import load_dotenv


# =============================
#  Streamlit Configuration
# =============================
st.set_page_config(
    page_title="MindMap AI Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =============================
#  Streamlit UI
# =============================
st.title("🧠 AI MindMap Generator")
st.markdown("Transform any public article URL into a structured **Mermaid.js** mind map.")

top_l, top_r = st.columns([6, 1])
with top_l:
    url_input = st.text_input(
        "Article URL",
        placeholder="e.g., https://hbr.org/2025/04/want-to-use-ai-as-a-career-coach-use-these-prompts"
    )
with top_r:
    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
    generate_button = st.button("🚀 Generate", disabled=not bool(url_input))

# Advanced settings
with st.expander("Advanced settings", expanded=False):
    adv_l, adv_r = st.columns(2)
    with adv_l:
        agent_timeout_sec = st.number_input("Agent timeout (seconds)", 30, 1800, 480, 30)
    with adv_r:
        agent_max_iter = st.number_input("Agent max iterations", 1, 20, 6, 1)

st.markdown("---")
st.header("💡 Generated Output")
output_placeholder = st.empty()

if generate_button:
    if not is_valid_url(url_input):
        output_placeholder.error("❌ Invalid URL. Please enter a valid https:// link.")
    else:
        with output_placeholder.container():
            st.info(f"Processing: {url_input}")
            with st.spinner("Working... this may take a minute..."):

                try:
                    result = generate_content(url_input, int(agent_timeout_sec), int(agent_max_iter))

                    if result.startswith("ERROR"):
                        st.error(result)
                    else:
                        clean_code = result
                        if "```mermaid" in clean_code:
                            start = clean_code.find("```mermaid") + 10
                            end = clean_code.find("```", start)
                            if end != -1:
                                clean_code = clean_code[start:end].strip()
                        elif "```" in clean_code:
                            clean_code = clean_code.replace("```", "").strip()

                        if not clean_code.startswith("graph"):
                            clean_code = "graph TD\n" + clean_code

                        st.success("✅ MindMap generated successfully!")

                        st.subheader("1. Mermaid.js Code")
                        st.code(clean_code, language="mermaid")

                        st.download_button(
                            "⬇️ Download Mermaid Code (.md)",
                            data=f"```mermaid\n{clean_code}\n```",
                            file_name="mindmap.md",
                            mime="text/markdown",
                            use_container_width=True
                        )

                        st.subheader("2. Visualization")
                        mermaid_html = f"""
                        <!DOCTYPE html>
                        <html>
                          <head>
                            <script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.0/dist/mermaid.min.js"></script>
                            <script>
                              mermaid.initialize({{ startOnLoad: true }});
                            </script>
                          </head>
                          <body>
                            <div class="mermaid">
                            {clean_code}
                            </div>
                          </body>
                        </html>
                        """
                        st.components.v1.html(mermaid_html, height=600, scrolling=True)

                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
                    traceback.print_exc()

st.markdown("---")
with st.expander("ℹ️ How it works"):
    st.markdown("""
    **Workflow:**
    1. Paste a URL from a public article.
    2. The AI agents will:
        - Scrape the article
        - Extract a structured topic hierarchy
        - Convert it into a styled Mermaid.js mind map
    3. You can view and download the code or render it directly.
    """)
