"""Streamlit chat interface for the Road Rules RAG Assistant."""
from __future__ import annotations

import io

import streamlit as st

import api_client

st.set_page_config(page_title="Road Rules Assistant", page_icon="🚦", layout="centered")

EXAMPLES = [
    "What is the BAC limit for drivers under 21?",
    "What must you do when a school bus flashes its red lights?",
    "How many seconds of following distance should I keep?",
    "What does a solid red traffic signal mean?",
]


# --------------------------------------------------------------------------- state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None


@st.cache_data(ttl=30, show_spinner=False)
def get_health() -> tuple[dict | None, str | None]:
    try:
        return api_client.health(), None
    except api_client.APIError as exc:
        return None, str(exc)


health, health_error = get_health()


# ------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🚦 Road Rules")
    st.caption("Answers grounded in official US state driver handbooks.")

    states = health.get("states", []) if health else []
    chosen = st.selectbox(
        "State handbook",
        ["All states"] + states,
        help="Filtering to one state prevents answers that mix rules from different states.",
    )
    selected_state = None if chosen == "All states" else chosen

    if selected_state is None and states:
        st.warning(
            "Searching all states. Rules genuinely differ between them - pick a state "
            "for a reliable answer.",
            icon="⚠️",
        )

    st.divider()
    st.subheader("Ask about a sign")
    uploaded = st.file_uploader(
        "Upload a road-sign photo", type=["jpg", "jpeg", "png", "webp", "bmp"]
    )
    if uploaded is not None:
        st.image(uploaded, caption=uploaded.name, use_container_width=True)
        if health and not health.get("yolo_loaded"):
            st.info("The vision model is not loaded on the backend.", icon="ℹ️")
        elif st.button("Identify sign and look up the rule", use_container_width=True):
            st.session_state.pending = {
                "kind": "image",
                "bytes": uploaded.getvalue(),
                "name": uploaded.name,
                "mime": uploaded.type,
            }
            st.rerun()

    st.divider()
    if health_error:
        st.error("Backend offline", icon="🔌")
    elif health:
        st.success(f"{health['chunks']:,} chunks indexed", icon="✅")
        st.caption(f"LLM: `{health['llm_model']}`")
        st.caption(f"Embeddings: `{health['embedding_model']}`")
        st.caption(f"Vision: {'loaded' if health['yolo_loaded'] else 'not loaded'}")
        if not health.get("llm_reachable", True):
            st.warning("Ollama is not reachable from the backend.", icon="⚠️")
    st.caption(f"API: `{api_client.API_BASE_URL}`")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------- main
st.title("Road Rules Assistant")

if health_error:
    st.error(health_error, icon="🔌")
    st.stop()

if not st.session_state.messages:
    st.caption("Ask a driving-rules question, or upload a sign photo from the sidebar.")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, use_container_width=True, key=f"ex{i}"):
            st.session_state.pending = {"kind": "text", "question": example}
            st.rerun()


def render(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("image"):
            st.image(message["image"], width=280)
        st.markdown(message["content"])
        if message.get("detections"):
            chips = "  ".join(
                f"`{d['label']}` {d['confidence']:.0%}" for d in message["detections"]
            )
            st.caption(f"Detected: {chips}")
        if message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])})"):
                for s in message["sources"]:
                    st.markdown(f"- {s}")


for message in st.session_state.messages:
    render(message)


# ------------------------------------------------------------------------- input
typed = st.chat_input("Ask about a driving rule...")
if typed:
    st.session_state.pending = {"kind": "text", "question": typed}
    st.rerun()


# ------------------------------------------------------------ handle pending work
pending = st.session_state.pending
if pending:
    st.session_state.pending = None

    if pending["kind"] == "text":
        user_message = {"role": "user", "content": pending["question"]}
    else:
        user_message = {
            "role": "user",
            "content": "_Uploaded a sign photo._",
            "image": pending["bytes"],
        }
    st.session_state.messages.append(user_message)
    render(user_message)

    with st.chat_message("assistant"):
        spinner_text = (
            "Detecting the sign and searching the handbooks..."
            if pending["kind"] == "image"
            else "Searching the handbooks..."
        )
        with st.spinner(spinner_text):
            try:
                if pending["kind"] == "text":
                    result = api_client.ask(pending["question"], selected_state)
                else:
                    result = api_client.ask_image(
                        pending["bytes"], pending["name"], pending["mime"], selected_state
                    )
                error = None
            except api_client.APIError as exc:
                result, error = None, str(exc)

        if error:
            st.error(error, icon="⚠️")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ {error}"}
            )
        else:
            st.markdown(result["answer"])
            if result.get("detections"):
                chips = "  ".join(
                    f"`{d['label']}` {d['confidence']:.0%}" for d in result["detections"]
                )
                st.caption(f"Detected: {chips}")
            if result.get("sources"):
                with st.expander(f"Sources ({len(result['sources'])})"):
                    for s in result["sources"]:
                        st.markdown(f"- {s}")
            else:
                st.caption("No sources - the assistant did not find this in the handbooks.")

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "detections": result.get("detections", []),
            })
