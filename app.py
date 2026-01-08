"""
FastAPI RAG Application - Streamlit UI
Slick, modern chat interface for querying FastAPI documentation.
"""

import streamlit as st
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from retriever import retrieve_context, search_documents
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

# Slick Modern CSS
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main .block-container {
        padding: 1rem 2rem 2rem 2rem;
        max-width: 1400px;
    }
    
    /* Dark header */
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
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 0.5rem;
    }
    
    /* Source cards */
    .source-card {
        background: #1e1e2e;
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 3px solid #63b3ab;
        padding: 0.9rem 1.1rem;
        margin: 0.4rem 0;
        border-radius: 0 10px 10px 0;
        transition: all 0.2s ease;
    }
    .source-card:hover {
        background: #252537;
        border-left-color: #7ec8c0;
    }
    .source-meta {
        color: #63b3ab;
        font-size: 0.72rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .source-text {
        color: rgba(255,255,255,0.75);
        font-size: 0.85rem;
        line-height: 1.5;
    }
    
    /* Metric cards */
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.3rem;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        margin-top: 0.4rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: #63b3ab;
        color: #1a1a2e;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.1rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #7ec8c0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255,255,255,0.02);
        padding: 0.4rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px;
        color: rgba(255,255,255,0.5);
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #63b3ab;
        color: #1a1a2e;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.02);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #1a1a2e; }
    ::-webkit-scrollbar-thumb { background: #3d3d5c; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #63b3ab; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None

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
    """Extract sources from tool messages in the agent response."""
    sources = []
    for msg in messages:
        if hasattr(msg, "type") and msg.type == "tool":
            content = getattr(msg, "content", "")
            if content and "[Relevance:" in content:
                chunks = content.split("\n\n---\n\n")
                for chunk in chunks:
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

# Detect if running on Streamlit Cloud
import os

IS_CLOUD = (
    os.getenv("STREAMLIT_SHARING_MODE") or os.getenv("STREAMLIT_SERVER_PORT") == "8501"
)

# Hero Header
st.markdown(
    """
<div class="hero-section">
    <h1>⚡ FastAPI Docs AI</h1>
    <p>Ask anything about FastAPI — powered by RAG with real-time documentation retrieval</p>
    <span class="hero-badge">🚀 Pinecone + LangChain</span>
</div>
""",
    unsafe_allow_html=True,
)

# Tabs - only show evaluation tab when running locally
if IS_CLOUD:
    tab_chat = st.container()
else:
    tab_chat, tab_eval = st.tabs(["💬 Ask Questions", "📊 Evaluate Responses"])

with tab_chat:
    col_main, col_side = st.columns([3, 1], gap="large")

    with col_main:
        # Chat container
        chat_container = st.container()

        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(
                    msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"
                ):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and msg.get("sources"):
                        with st.expander(f"📚 {len(msg['sources'])} Sources Retrieved"):
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
                                    "question": prompt,
                                }
                            )
                            if sources:
                                with st.expander(
                                    f"📚 {len(sources)} Sources Retrieved",
                                    expanded=False,
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
            "Path parameters vs query params",
            "Dependency injection",
            "Request validation",
        ]
        for ex in examples:
            if st.button(f"→ {ex}", key=ex, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()

        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.evaluation_results = None
            st.rerun()

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.caption(
            "This AI assistant retrieves information from FastAPI documentation using RAG (Retrieval-Augmented Generation)."
        )

# Only show evaluation tab when running locally
if not IS_CLOUD:
    with tab_eval:
        st.markdown("### 📊 RAGAS Evaluation")
        st.caption("Measure faithfulness and detect hallucinations")

        assistant_msgs = [
            m for m in st.session_state.messages if m.get("role") == "assistant"
        ]

        if not assistant_msgs:
            st.info("💬 Ask a question first, then evaluate responses here.")
        else:
            options = [
                f"Response {i+1}: {m['content'][:50]}..."
                for i, m in enumerate(assistant_msgs)
            ]
            idx = st.selectbox(
                "Select response:",
                range(len(options)),
                format_func=lambda x: options[x],
            )

            selected = assistant_msgs[idx]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Question**")
                st.info(selected.get("question", "N/A"))
            with col2:
                st.markdown("**Sources**")
                st.success(f"{len(selected.get('sources', []))} chunks retrieved")

            st.markdown("**Answer Preview**")
            st.markdown(f"> {selected['content'][:300]}...")

            if st.button(
                "🔬 Run RAGAS Evaluation", type="primary", use_container_width=True
            ):
                sources = selected.get("sources", [])
                if not sources:
                    st.warning("No sources available.")
                else:
                    with st.spinner("⏳ Evaluating..."):
                        try:
                            from evaluation import evaluate_response

                            results = evaluate_response(
                                question=selected.get("question", ""),
                                answer=selected["content"],
                                contexts=[s["text"] for s in sources],
                            )
                            st.session_state.evaluation_results = results
                        except Exception as e:
                            st.error(f"Error: {e}")

            if st.session_state.evaluation_results:
                st.markdown("---")
                results = st.session_state.evaluation_results
                cols = st.columns(2)

                metrics = {
                    "faithfulness": (
                        "🎯 Faithfulness",
                        "Grounded in sources (no hallucinations)",
                    ),
                    "answer_relevancy": (
                        "✅ Relevancy",
                        "Directly answers the question",
                    ),
                }

                for col, (key, (label, desc)) in zip(cols, metrics.items()):
                    if key in results:
                        score = results[key]
                        with col:
                            if score is not None and isinstance(score, (int, float)):
                                color = (
                                    "#22c55e"
                                    if score >= 0.7
                                    else "#f97316" if score >= 0.4 else "#ef4444"
                                )
                                display_val = f"{score:.2f}"
                            else:
                                color = "#888888"
                                display_val = "N/A"
                            st.markdown(
                                f"""
                            <div class="metric-card">
                                <div class="metric-value" style="color: {color}">{display_val}</div>
                                <div class="metric-label">{label}</div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )
                            st.caption(desc)
