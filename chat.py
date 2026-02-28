"""
NASA RAG Chat UI — Streamlit interface.
Connects rag_client + llm_client + ragas_evaluator together.
"""

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

import rag_client
import llm_client
import ragas_evaluator

# Page config — must be FIRST streamlit command
st.set_page_config(
    page_title="NASA Mission Intelligence",
    page_icon="🚀",
    layout="wide"
)

def init_session_state():
    """
    Streamlit reruns entire script on every interaction.
    Session state PERSISTS data between reruns.
    
    Without this: chat history disappears after every message!
    With this:    history stays throughout the session.
    """
    defaults = {
        "messages":        [],    # chat history
        "current_backend": None,  # which ChromaDB collection
        "last_scores":     None,  # last RAGAS scores
        "last_contexts":   [],    # last retrieved chunks
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
            
## ── MAIN APP FUNCTION ───────────────────────────────────────────────  
          
def main():
    init_session_state()

    st.title("🚀 NASA Mission Intelligence Chat")
    st.markdown(
        "Ask questions about **Apollo 11, Apollo 13, "
        "and Challenger** missions."
    )

    # ── SIDEBAR ──────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Auto-discover ChromaDB backends
        with st.spinner("Finding collections..."):
            backends = rag_client.discover_chroma_backends()

        if not backends:
            st.error("No ChromaDB found! Run embedding_pipeline.py first.")
            st.code("python embedding_pipeline.py --data-path data_text")
            st.stop()

        # Backend selector dropdown
        st.subheader("📚 Document Collection")
        backend_options = {k: v["display_name"] for k, v in backends.items()}
        selected_key = st.selectbox(
            "Select Collection",
            options=list(backend_options.keys()),
            format_func=lambda x: backend_options[x]
        )
        selected_backend = backends[selected_key]

        # API Key input
        st.subheader("🔑 API Key")
        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            help="Your Claude API key"
        )
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        else:
            st.warning("Please enter your Anthropic API key")
            st.stop()

        # Retrieval settings
        st.subheader("🔍 Search Settings")
        n_docs = st.slider("Chunks to retrieve", 1, 10, 3)
        mission_filter = st.selectbox(
            "Filter by Mission",
            ["All", "apollo_11", "apollo_13", "challenger"]
        )
        mission_val = None if mission_filter == "All" else mission_filter

        # Evaluation toggle
        st.subheader("📊 Evaluation")
        enable_eval = st.checkbox("Enable RAGAS Scoring", value=True)

        # Show last scores in sidebar
        if st.session_state.last_scores and enable_eval:
            st.subheader("Last Response Scores")
            for metric, score in st.session_state.last_scores.items():
                if isinstance(score, float):
                    st.metric(
                        label=metric.replace("_", " ").title(),
                        value=f"{score:.3f}"
                    )
                    st.progress(float(score))

        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.last_scores = None
            st.rerun()

    # ── CONNECT TO CHROMADB ───────────────────────────────
    # Reset if user switched collection
    if st.session_state.current_backend != selected_key:
        st.session_state.current_backend = selected_key
        st.cache_resource.clear()

    with st.spinner("Connecting to ChromaDB..."):
        collection, success, error = rag_client.initialize_rag_system(
            selected_backend["directory"],
            selected_backend["collection_name"]
        )

    if not success:
        st.error(f"ChromaDB connection failed: {error}")
        st.stop()

    # ── DISPLAY CHAT HISTORY ──────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── CHAT INPUT ────────────────────────────────────────
    if prompt := st.chat_input("Ask about NASA missions..."):

        # Show user message
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):

            # Step 1: Retrieve chunks
            with st.spinner("🔍 Searching NASA documents..."):
                results = rag_client.retrieve_documents(
                    collection, prompt, n_docs, mission_val
                )

            # Step 2: Format context
            context = ""
            raw_docs = []
            if results and results.get("documents"):
                raw_docs  = results["documents"][0]
                raw_metas = results["metadatas"][0]
                raw_dists = results.get("distances", [[]])[0]
                context   = rag_client.format_context(
                    raw_docs, raw_metas, raw_dists
                )
                st.session_state.last_contexts = raw_docs

            # Step 3: Ask Claude
            with st.spinner("🤖 Claude is thinking..."):
                answer = llm_client.generate_response(
                    openai_key="",
                    user_message=prompt,
                    context=context,
                    conversation_history=st.session_state.messages[:-1]
                )

            # Show answer
            st.markdown(answer)

            # Step 4: RAGAS evaluation
            if enable_eval and raw_docs:
                with st.spinner("📊 Scoring with RAGAS..."):
                    scores = ragas_evaluator.evaluate_response_quality(
                        prompt, answer, raw_docs
                    )
                    st.session_state.last_scores = scores

                if "error" not in scores:
                    cols = st.columns(len(scores))
                    for col, (metric, score) in zip(cols, scores.items()):
                        col.metric(
                            label=metric.replace("_", " ").title(),
                            value=f"{score:.3f}"
                        )
                else:
                    st.warning(f"Evaluation: {scores['error']}")

            # Show retrieved context in expander
            with st.expander("📄 View Retrieved Context"):
                st.text(context if context else "No context retrieved.")

        # Save assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


if __name__ == "__main__":
    main()