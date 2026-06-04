"""
ui.py — Streamlit chat interface for the RAG chatbot.

Features:
  ✅ Chat interface with message history
  ✅ Streaming-style "thinking" indicator
  ✅ Image citations — figures from the paper shown inline
  ✅ Table citations — Markdown tables rendered inline
  ✅ Text citations — source passage expandable panels
  ✅ Equation citations — highlighted math content
  ✅ Sidebar with session info, pipeline metadata, and settings
  ✅ Debug mode toggle — shows rewritten query, route, validation
  ✅ Session management — new session button, end & save memory

Run with:
  uv run streamlit run ui.py

Make sure the API is running first:
  uv run uvicorn api:app --reload --port 8000
"""
from __future__ import annotations

import base64
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL     = "http://localhost:8000"
APP_TITLE   = "📖 RAG Chatbot — Attention Is All You Need"
PAGE_ICON   = "🤖"


# ── Page setup — must be the FIRST Streamlit call ─────────────────────────────
st.set_page_config(
    page_title = APP_TITLE,
    page_icon  = PAGE_ICON,
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main chat container */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}
.main-header h1 { margin: 0; font-size: 1.6rem; }
.main-header p  { margin: 0.3rem 0 0 0; opacity: 0.75; font-size: 0.9rem; }

/* User message bubble */
.user-bubble {
    background: #0f3460;
    color: white;
    padding: 0.9rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.5rem 0 0.5rem 20%;
    font-size: 0.95rem;
    line-height: 1.5;
}

/* Assistant message bubble */
.assistant-bubble {
    background: #1e1e2e;
    color: #e0e0e0;
    padding: 0.9rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.5rem 20% 0.5rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
    border-left: 3px solid #0f3460;
}

/* Citation cards */
.citation-card {
    background: #12122a;
    border: 1px solid #2d2d4e;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
}
.citation-header {
    color: #7ecfff;
    font-weight: 600;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
}
.citation-type-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
}
.badge-image    { background: #1a3a1a; color: #7fff7f; }
.badge-table    { background: #1a2a3a; color: #7fcfff; }
.badge-equation { background: #2a1a3a; color: #cf7fff; }
.badge-text     { background: #2a2a1a; color: #ffcf7f; }
.badge-web      { background: #3a1a1a; color: #ff7f7f; }

/* Image citation */
.figure-caption {
    text-align: center;
    font-size: 0.8rem;
    color: #aaa;
    margin-top: 0.4rem;
    font-style: italic;
}

/* Table citation */
.table-citation {
    overflow-x: auto;
    font-size: 0.8rem;
}

/* Sidebar stat */
.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid #2d2d4e;
    font-size: 0.82rem;
}
.stat-label { color: #888; }
.stat-value { color: #7ecfff; font-weight: 600; }

/* Thinking indicator */
.thinking {
    color: #888;
    font-style: italic;
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }

/* Route badge */
.route-rag        { color: #7fff7f; }
.route-web-search { color: #ffcf7f; }

/* Scrollable chat history */
.chat-history-container {
    max-height: 65vh;
    overflow-y: auto;
    padding-right: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
# Streamlit re-runs the entire script on every interaction.
# st.session_state persists values across re-runs.

def _init_session_state() -> None:
    """Initialise all session state variables on first load."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "user_id" not in st.session_state:
        st.session_state.user_id = "default_user"

    if "messages" not in st.session_state:
        # Each message: {"role": "user"|"assistant", "content": str,
        #                "cited_sources": list, "meta": dict}
        st.session_state.messages = []

    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

    if "last_meta" not in st.session_state:
        st.session_state.last_meta = {}

    if "api_reachable" not in st.session_state:
        st.session_state.api_reachable = None


_init_session_state()


# ── API helpers ───────────────────────────────────────────────────────────────

def _check_api() -> bool:
    """Check if the FastAPI backend is reachable."""
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def _send_query(query: str) -> Optional[Dict]:
    """
    Send a query to the API and return the response dict.
    Returns None on error.
    """
    try:
        r = httpx.post(
            f"{API_URL}/chat",
            json    = {
                "query":      query,
                "user_id":    st.session_state.user_id,
                "session_id": st.session_state.session_id,
            },
            timeout = 120.0,   # 2 minutes — 70b model can be slow
        )
        r.raise_for_status()
        return r.json()
    except httpx.TimeoutException:
        st.error("⏱️  Request timed out. The 70b model may be under load. Please try again.")
    except httpx.HTTPStatusError as e:
        st.error(f"❌  API error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        st.error(f"❌  Could not reach API: {e}")
    return None


def _end_session() -> None:
    """Tell the API to save long-term memory and reset local session."""
    try:
        httpx.post(
            f"{API_URL}/end-session",
            json    = {
                "user_id":    st.session_state.user_id,
                "session_id": st.session_state.session_id,
            },
            timeout = 30.0,
        )
    except Exception:
        pass  # best effort

    # Reset local state for the new session
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages   = []
    st.session_state.last_meta  = {}


# ── Citation rendering ────────────────────────────────────────────────────────

def _render_image_citation(source: Dict, index: int) -> None:
    """
    Render an image citation with:
    - Caption
    - The actual image (decoded from base64)
    - Expandable description text
    """
    caption  = source.get("caption", f"Figure (Doc {source['doc_number']})")
    b64_data = source.get("image_base64", "")
    content  = source.get("content", "")

    # Build a readable title for the expander
    section = source.get("section_header", "")
    page    = source.get("page_number", 0)
    title   = f"🖼️  {caption or 'Figure'}"
    if section: title += f"  —  {section}"
    if page:    title += f"  (p.{page})"

    with st.expander(title, expanded=(index == 0)):  # expand first citation by default
        if b64_data:
            # Decode base64 → PIL Image → display
            try:
                img_bytes = base64.b64decode(b64_data)
                image     = Image.open(BytesIO(img_bytes))
                st.image(
                    image,
                    caption    = caption or "Figure from the paper",
                    use_container_width = True,
                )
            except Exception:
                st.warning("⚠️  Could not render image.")
        else:
            st.info("🖼️  Image file not available locally. Description below:")

        # Show the description text (extracted by the vision model during ingestion)
        if content:
            # Extract just the description part from the composite chunk
            desc_match = __import__("re").search(
                r"Description:\s*(.*?)(?:\(Section:|$)", content, __import__("re").DOTALL
            )
            if desc_match:
                st.markdown(f"**Description:** {desc_match.group(1).strip()}")
            else:
                with st.container():
                    st.text_area(
                        "Full content",
                        value   = content[:500] + ("…" if len(content) > 500 else ""),
                        height  = 100,
                        disabled= True,
                        key     = f"img_content_{index}_{source['doc_number']}",
                    )


def _render_table_citation(source: Dict, index: int) -> None:
    """
    Render a table citation with:
    - Caption
    - The Markdown table rendered as a proper HTML table
    - Expandable full content
    """
    caption  = source.get("caption", f"Table (Doc {source['doc_number']})")
    table_md = source.get("table_markdown", "")
    section  = source.get("section_header", "")
    page     = source.get("page_number", 0)

    title = f"📊  {caption or 'Table'}"
    if section: title += f"  —  {section}"
    if page:    title += f"  (p.{page})"

    with st.expander(title, expanded=(index == 0)):
        if table_md:
            # Streamlit renders Markdown tables natively
            st.markdown(table_md)
        else:
            content = source.get("content", "")
            if content:
                # Try to extract the table portion from the composite chunk
                import re
                tbl_match = re.search(
                    r"\[TABLE CONTENT\]\s*(.*?)(?=\[|$)", content, re.DOTALL
                )
                if tbl_match:
                    st.markdown(tbl_match.group(1).strip())
                else:
                    st.text(content[:600])


def _render_text_citation(source: Dict, index: int) -> None:
    """Render a text citation as a collapsible passage."""
    section = source.get("section_header", "")
    page    = source.get("page_number", 0)
    content = source.get("content", "")

    title = f"📄  Doc {source['doc_number']}"
    if section: title += f"  —  {section}"
    if page:    title += f"  (p.{page})"

    with st.expander(title, expanded=False):
        # Show a clean snippet (first 400 chars)
        snippet = content[:400] + ("…" if len(content) > 400 else "")
        st.markdown(f"*{snippet}*")


def _render_equation_citation(source: Dict, index: int) -> None:
    """Render an equation citation with surrounding context."""
    section = source.get("section_header", "")
    page    = source.get("page_number", 0)
    content = source.get("content", "")

    title = f"∑  Equation  —  Doc {source['doc_number']}"
    if section: title += f"  ({section})"
    if page:    title += f"  (p.{page})"

    with st.expander(title, expanded=False):
        import re
        eq_match = re.search(
            r"\[EQUATION CONTENT\]\s*(.*?)(?=\[|$)", content, re.DOTALL
        )
        if eq_match:
            eq_text = eq_match.group(1).strip()
            st.code(eq_text, language=None)
        else:
            st.markdown(f"```\n{content[:300]}\n```")


def _render_web_citation(source: Dict, index: int) -> None:
    """Render a web search result citation."""
    url     = source.get("source_url", "")
    content = source.get("content", "")

    title = f"🌐  Web Result {index + 1}"
    if url:
        domain = url.split("/")[2] if "//" in url else url
        title += f"  —  {domain}"

    with st.expander(title, expanded=False):
        if url:
            st.markdown(f"🔗 [{url}]({url})")
        snippet = content[:400] + ("…" if len(content) > 400 else "")
        st.markdown(f"*{snippet}*")


def _render_citations(sources: List[Dict]) -> None:
    """
    Main citation renderer — dispatches each source to the right renderer
    based on its element_type.
    """
    if not sources:
        return

    # Group sources by type for organised display
    images    = [(i, s) for i, s in enumerate(sources) if s["element_type"] == "image"]
    tables    = [(i, s) for i, s in enumerate(sources) if s["element_type"] == "table"]
    equations = [(i, s) for i, s in enumerate(sources) if s["element_type"] == "equation"]
    texts     = [(i, s) for i, s in enumerate(sources) if s["element_type"] == "text"]
    web       = [(i, s) for i, s in enumerate(sources) if s.get("source_url")]

    # Render images first (most visual)
    if images:
        st.markdown("**🖼️ Figure References**")
        for idx, (i, source) in enumerate(images):
            _render_image_citation(source, idx)

    # Tables next
    if tables:
        st.markdown("**📊 Table References**")
        for idx, (i, source) in enumerate(tables):
            _render_table_citation(source, idx)

    # Equations
    if equations:
        st.markdown("**∑ Equation References**")
        for idx, (i, source) in enumerate(equations):
            _render_equation_citation(source, idx)

    # Text passages
    if texts:
        st.markdown("**📄 Text Passages**")
        for idx, (i, source) in enumerate(texts):
            _render_text_citation(source, idx)

    # Web results
    if web:
        st.markdown("**🌐 Web Sources**")
        for idx, (i, source) in enumerate(web):
            _render_web_citation(source, idx)


# ── Message rendering ─────────────────────────────────────────────────────────

def _render_message(msg: Dict) -> None:
    """
    Render one complete message (user or assistant) in the chat area.
    Assistant messages include the citations panel below the answer text.
    """
    role    = msg["role"]
    content = msg["content"]

    if role == "user":
        st.markdown(
            f'<div class="user-bubble">👤 {content}</div>',
            unsafe_allow_html=True,
        )

    else:
        # Assistant message
        st.markdown(
            f'<div class="assistant-bubble">🤖 {content}</div>',
            unsafe_allow_html=True,
        )

        # Show debug metadata if enabled
        meta = msg.get("meta", {})
        if st.session_state.debug_mode and meta:
            with st.expander("🔧 Debug info", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Route",      meta.get("route", "—"))
                col2.metric("Validation", meta.get("validation_result", "—"))
                col3.metric("Retries",    meta.get("retry_count", 0))
                col4.metric("Sources",    len(msg.get("cited_sources", [])))
                if meta.get("rewritten_query"):
                    st.caption(f"Rewritten query: *{meta['rewritten_query']}*")

        # Render citations
        cited = msg.get("cited_sources", [])
        if cited:
            st.markdown("---")
            _render_citations(cited)


def _render_all_messages() -> None:
    """Render the full chat history."""
    for msg in st.session_state.messages:
        _render_message(msg)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    """Render the sidebar with user settings, session info, and controls."""
    with st.sidebar:
        # ── App header ────────────────────────────────────────────────────
        st.markdown("## 🤖 RAG Chatbot")
        st.caption("Powered by LangGraph · Groq · Qdrant")
        st.divider()

        # ── User identity ─────────────────────────────────────────────────
        st.markdown("### 👤 User")
        new_user = st.text_input(
            "Username",
            value       = st.session_state.user_id,
            key         = "user_id_input",
            help        = "Used for long-term memory — your past sessions are remembered",
            label_visibility = "collapsed",
            placeholder = "Enter your username",
        )
        if new_user != st.session_state.user_id:
            st.session_state.user_id = new_user

        st.caption(f"Session: `{st.session_state.session_id[:12]}…`")
        st.divider()

        # ── API Status ─────────────────────────────────────────────────────
        st.markdown("### 🔌 API Status")
        if st.button("Check Connection", use_container_width=True):
            st.session_state.api_reachable = _check_api()

        if st.session_state.api_reachable is True:
            st.success("✅  API is running")
        elif st.session_state.api_reachable is False:
            st.error(
                "❌  API not reachable\n\n"
                "Start it with:\n"
                "```\nuv run uvicorn api:app --reload --port 8000\n```"
            )
        else:
            st.info("Click 'Check Connection' to test")

        st.divider()

        # ── Last query stats ───────────────────────────────────────────────
        meta = st.session_state.last_meta
        if meta:
            st.markdown("### 📊 Last Query Info")

            route = meta.get("route", "—")
            route_color = "🟢" if route == "rag" else "🟡"
            st.markdown(f"{route_color} **Route:** `{route}`")

            validation = meta.get("validation_result", "—")
            val_color  = "✅" if validation == "valid" else "⚠️"
            st.markdown(f"{val_color} **Validation:** `{validation}`")

            retries = meta.get("retry_count", 0)
            st.markdown(f"🔄 **Retries:** `{retries}`")

            n_sources = meta.get("n_sources", 0)
            st.markdown(f"📚 **Sources used:** `{n_sources}`")

            if st.session_state.debug_mode:
                rq = meta.get("rewritten_query", "")
                if rq:
                    st.caption(f"**Rewritten:** {rq}")

            st.divider()

        # ── Quick examples ─────────────────────────────────────────────────
        st.markdown("### 💡 Example Questions")
        examples = [
            "How does multi-head attention work?",
            "What does Figure 1 show?",
            "Compare model sizes in Table 2",
            "Why use positional encoding?",
            "What is the scaled dot-product attention formula?",
            "What BLEU score did the big transformer achieve?",
        ]
        for example in examples:
            if st.button(
                example,
                key              = f"ex_{example[:20]}",
                use_container_width = True,
            ):
                # Inject the example as if the user typed it
                st.session_state["prefill_query"] = example
                st.rerun()

        st.divider()

        # ── Settings ───────────────────────────────────────────────────────
        st.markdown("### ⚙️ Settings")
        st.session_state.debug_mode = st.toggle(
            "Debug mode",
            value = st.session_state.debug_mode,
            help  = "Shows rewritten query, route, validation result per message",
        )

        st.divider()

        # ── Session controls ───────────────────────────────────────────────
        st.markdown("### 🗂️ Session")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save & New", use_container_width=True, help="Save memory and start a fresh session"):
                _end_session()
                st.success("Session saved!")
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True, help="Clear chat (memory is NOT saved)"):
                st.session_state.messages  = []
                st.session_state.last_meta = {}
                st.rerun()


# ── Main UI ───────────────────────────────────────────────────────────────────

def main() -> None:
    """Main UI render function — called on every Streamlit interaction."""

    # ── Sidebar ────────────────────────────────────────────────────────────
    _render_sidebar()

    # ── Page header ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>📖 RAG Chatbot — Attention Is All You Need</h1>
        <p>Ask anything about the 2017 Transformer paper · Images, tables, and equations shown inline</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Chat history ───────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center; padding:3rem; color:#555;">
                <div style="font-size:3rem">📚</div>
                <div style="font-size:1.1rem; margin-top:1rem">
                    Ask a question about the Transformer paper.
                </div>
                <div style="font-size:0.85rem; margin-top:0.5rem; color:#444;">
                    Figures, tables, and equations from the paper will appear as citations.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            _render_all_messages()

    # ── Chat input ─────────────────────────────────────────────────────────
    # Handle prefilled query from sidebar example buttons
    prefill = st.session_state.pop("prefill_query", "")

    user_input = st.chat_input(
        placeholder = "Ask anything about the paper…  (e.g. 'How does multi-head attention work?')",
    )

    # Use prefill if available, otherwise use typed input
    query = prefill or user_input

    if query:
        # 1. Add user message to history immediately so it shows up
        st.session_state.messages.append({
            "role":          "user",
            "content":       query,
            "cited_sources": [],
            "meta":          {},
        })

        # 2. Show thinking indicator
        with st.spinner("🤔  Thinking…  (routing → retrieving → grading → generating)"):
            response = _send_query(query)

        if response:
            # 3. Add assistant message with full citations
            sources = response.get("cited_sources", [])
            meta    = {
                "route":             response.get("route", ""),
                "rewritten_query":   response.get("rewritten_query", ""),
                "validation_result": response.get("validation_result", ""),
                "retry_count":       response.get("retry_count", 0),
                "n_sources":         len(sources),
            }

            st.session_state.messages.append({
                "role":          "assistant",
                "content":       response.get("answer", "No answer generated."),
                "cited_sources": sources,
                "meta":          meta,
            })

            # Update sidebar stats
            st.session_state.last_meta = meta

        # 4. Rerun to refresh the chat display
        st.rerun()


if __name__ == "__main__":
    main()