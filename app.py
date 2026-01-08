"""
FastAPI RAG Application - Streamlit UI
Professional chat interface for querying FastAPI documentation.
"""

import streamlit as st
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from retriever import retrieve_context, search_documents
from config import LLM_MODEL, SYSTEM_PROMPT

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FastAPI Documentation Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional CSS
st.markdown(
    """
<style>
    .main .block-container { padding: 2rem 3rem; max-width: 1200px; }
    
    .header-gradient {
        background: linear-gradient(135deg, #009688 0%, #00796b 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .header-gradient h1 { margin: 0; font-size: 1.8rem; }
    .header-gradient p { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }
    
    .source-card {
        background: #fafafa;
        border-left: 3px solid #009688;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }
    .source-meta { color: #666; font-size: 0.8rem; margin-bottom: 0.4rem; }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #666; }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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
    llm = ChatOllama(model=LLM_MODEL)
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

# Header
st.markdown(
    """
<div class="header-gradient">
    <h1>⚡ FastAPI Documentation Assistant</h1>
    <p>Get accurate answers about FastAPI from the official documentation</p>
</div>
""",
    unsafe_allow_html=True,
)

# Tabs
tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation"])

with tab_chat:
    col_main, col_side = st.columns([3, 1])

    with col_main:
        # Chat history
        for msg in st.session_state.messages:
            avatar = "👤" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander(f"📚 View {len(msg['sources'])} Sources"):
                        for src in msg["sources"]:
                            st.markdown(
                                f"""
                            <div class="source-card">
                                <div class="source-meta">{src['meta']}</div>
                                {src['text'][:300]}...
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

        # Chat input
        if prompt := st.chat_input("Ask about FastAPI..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Searching documentation..."):
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
                                    f"📚 View {len(sources)} Sources", expanded=False
                                ):
                                    for src in sources:
                                        st.markdown(
                                            f"""
                                        <div class="source-card">
                                            <div class="source-meta">{src['meta']}</div>
                                            {src['text'][:300]}...
                                        </div>
                                        """,
                                            unsafe_allow_html=True,
                                        )
                        else:
                            st.error("Unable to generate response. Please try again.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with col_side:
        st.markdown("### 💡 Example Questions")
        examples = [
            "What is FastAPI?",
            "How do I create a POST endpoint?",
            "Explain path parameters",
            "What is dependency injection?",
        ]
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state.pending_question = ex
                st.rerun()

        st.divider()
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.evaluation_results = None
            st.rerun()

with tab_eval:
    st.markdown("### 📊 Response Evaluation")
    st.caption("Evaluate responses for hallucinations using RAGAS metrics")

    assistant_msgs = [
        m for m in st.session_state.messages if m.get("role") == "assistant"
    ]

    if not assistant_msgs:
        st.info("💬 Ask a question first, then evaluate the response here.")
    else:
        options = [
            f"Response {i+1}: {m['content'][:40]}..."
            for i, m in enumerate(assistant_msgs)
        ]
        idx = st.selectbox(
            "Select response:", range(len(options)), format_func=lambda x: options[x]
        )

        selected = assistant_msgs[idx]
        st.markdown(f"**Question:** {selected.get('question', 'N/A')}")
        st.markdown(f"**Answer:** {selected['content'][:200]}...")
        st.markdown(f"**Sources:** {len(selected.get('sources', []))} chunks")

        if st.button("🔬 Run RAGAS Evaluation", type="primary"):
            sources = selected.get("sources", [])
            if not sources:
                st.warning("No sources available for evaluation.")
            else:
                with st.spinner("Evaluating (this may take 1-2 minutes)..."):
                    try:
                        from evaluation import evaluate_response

                        results = evaluate_response(
                            question=selected.get("question", ""),
                            answer=selected["content"],
                            contexts=[s["text"] for s in sources],
                        )
                        st.session_state.evaluation_results = results
                    except Exception as e:
                        st.error(f"Evaluation error: {e}")

        if st.session_state.evaluation_results:
            results = st.session_state.evaluation_results
            cols = st.columns(2)

            metrics = {
                "faithfulness": ("🎯 Faithfulness", "Grounded in sources?"),
                "answer_relevancy": ("✅ Relevancy", "Answers the question?"),
            }

            for col, (key, (label, desc)) in zip(cols, metrics.items()):
                if key in results:
                    score = results[key]
                    if isinstance(score, (int, float)):
                        color = (
                            "#22c55e"
                            if score >= 0.7
                            else "#f97316" if score >= 0.4 else "#ef4444"
                        )
                        with col:
                            st.markdown(
                                f"""
                            <div class="metric-card">
                                <div class="metric-value" style="color: {color}">{score:.2f}</div>
                                <div class="metric-label">{label}</div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )
                            st.caption(desc)
