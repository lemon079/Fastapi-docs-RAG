"""
FastAPI RAG Application - Streamlit UI
Simple chat interface for querying FastAPI documentation.
"""

import streamlit as st
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from retriever import retrieve_context
from config import LLM_MODEL, SYSTEM_PROMPT, OLLAMA_BASE_URL

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FastAPI Docs AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Muted teal CSS
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main .block-container {
        padding: 1rem 2rem 2rem 2rem;
        max-width: 1400px;
    }
    
    .hero-section {
        background: #1a1a2e;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero-section h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #fff;
    }
    .hero-section p {
        margin: 0.4rem 0 0 0;
        color: rgba(255,255,255,0.6);
        font-size: 0.95rem;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99, 179, 171, 0.2);
        color: #63b3ab;
        padding: 0.25rem 0.7rem;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-top: 0.8rem;
        border: 1px solid rgba(99, 179, 171, 0.3);
    }
    
    .stChatMessage {
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 0.5rem;
    }
    
    .source-card {
        background: #1e1e2e;
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 3px solid #63b3ab;
        padding: 0.9rem 1.1rem;
        margin: 0.4rem 0;
        border-radius: 0 10px 10px 0;
    }
    .source-meta {
        color: #63b3ab;
        font-size: 0.72rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
    }
    .source-text {
        color: rgba(255,255,255,0.75);
        font-size: 0.85rem;
        line-height: 1.5;
    }
    
    .stButton > button {
        background: #63b3ab;
        color: #1a1a2e;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.1rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #7ec8c0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# CACHED RESOURCES
# ============================================================


@st.cache_resource
def get_agent():
    """Initialize and cache the RAG agent."""
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    return create_agent(llm, [retrieve_context], system_prompt=SYSTEM_PROMPT)


agent = get_agent()

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def parse_sources_from_tool_output(messages: list) -> list:
    """Extract sources from tool messages."""
    sources = []
    for msg in messages:
        if hasattr(msg, "type") and msg.type == "tool":
            content = getattr(msg, "content", "")
            if content and "[Relevance:" in content:
                for chunk in content.split("\n\n---\n\n"):
                    if "[Relevance:" in chunk:
                        try:
                            meta = chunk.split("]")[0].replace("[", "")
                            text = chunk.split("]", 1)[1].strip()
                            sources.append({"meta": meta, "text": text})
                        except:
                            pass
    return sources


# ============================================================
# UI LAYOUT
# ============================================================

# Hero Header
st.markdown(
    """
<div class="hero-section">
    <h1>⚡ FastAPI Docs AI</h1>
    <p>Ask anything about FastAPI — powered by RAG</p>
    <span class="hero-badge">🚀 Pinecone + LangChain</span>
</div>
""",
    unsafe_allow_html=True,
)

# Main layout
col_main, col_side = st.columns([3, 1], gap="large")

with col_main:
    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(
            msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"
        ):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📚 {len(msg['sources'])} Sources"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(
                            f"""
                        <div class="source-card">
                            <div class="source-meta">Source {i} • {src['meta']}</div>
                            <div class="source-text">{src['text'][:250]}...</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

    # Chat input
    if prompt := st.chat_input("💬 Ask about FastAPI..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Searching docs..."):
                try:
                    response = ""
                    sources = []

                    for event in agent.stream(
                        {"messages": [{"role": "user", "content": prompt}]},
                        stream_mode="values",
                        config={"recursion_limit": 25},
                    ):
                        sources = parse_sources_from_tool_output(event["messages"])
                        for msg in event["messages"]:
                            if hasattr(msg, "type") and msg.type == "ai":
                                if hasattr(msg, "content") and msg.content:
                                    if not getattr(msg, "tool_calls", None):
                                        response = msg.content

                    if response:
                        st.markdown(response)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response,
                                "sources": sources,
                            }
                        )
                        if sources:
                            with st.expander(
                                f"📚 {len(sources)} Sources", expanded=False
                            ):
                                for i, src in enumerate(sources, 1):
                                    st.markdown(
                                        f"""
                                    <div class="source-card">
                                        <div class="source-meta">Source {i} • {src['meta']}</div>
                                        <div class="source-text">{src['text'][:250]}...</div>
                                    </div>
                                    """,
                                        unsafe_allow_html=True,
                                    )
                    else:
                        st.error("Unable to generate response.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with col_side:
    st.markdown("### 💡 Try These")
    examples = [
        "What is FastAPI?",
        "Create a POST endpoint",
        "Path parameters",
        "Dependency injection",
    ]
    for ex in examples:
        if st.button(f"→ {ex}", key=ex, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("AI assistant for FastAPI documentation using RAG.")
