import os
import tempfile

import streamlit as st

# pyrefly: ignore [missing-import]
from llm import ask_llm

from rag.document_rag import (
    process_pdf,
    set_active_document,
    clear_active_document,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Anil AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "document" not in st.session_state:

    st.session_state.document = None


if "suggested_prompt" not in st.session_state:

    st.session_state.suggested_prompt = None


# ============================================================
# RESTORE ACTIVE DOCUMENT
# ============================================================

if st.session_state.document:

    set_active_document(
        st.session_state
        .document["name"]
    )


# ============================================================
# CLEAN UI CSS
# ============================================================

st.markdown(
    """
<style>

/* ==========================================================
   MAIN
========================================================== */

.block-container {

    max-width: 1000px;

    padding-top: 2rem;

    padding-bottom: 7rem;
}


/* ==========================================================
   SIDEBAR
========================================================== */

[data-testid="stSidebar"] {

    min-width: 290px;

    max-width: 290px;
}


/* ==========================================================
   BUTTONS
========================================================== */

.stButton > button {

    border-radius: 10px;

    min-height: 42px;
}


/* ==========================================================
   CHAT
========================================================== */

[data-testid="stChatMessage"] {

    padding-top: 1rem;

    padding-bottom: 1rem;
}


/* ==========================================================
   INPUT
========================================================== */

[data-testid="stChatInput"] {

    border-radius: 14px;
}


/* ==========================================================
   TYPOGRAPHY
========================================================== */

h1, h2, h3 {

    letter-spacing: -0.02em;
}


/* ==========================================================
   FOOTER
========================================================== */

footer {

    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖⚡ Nexora AI")

    st.caption(
        "Agentic AI assistant powered by tool calling, "
        "real-time web search, and Hybrid RAG."
    )

    # ========================================================
    # NEW CHAT
    # ========================================================

    if st.button(
        "＋ New chat",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


    st.divider()


    # ========================================================
    # STATUS
    # ========================================================

    st.caption(
        "STATUS"
    )


    st.success(
        "● Agent online"
    )


    st.caption(
        "Groq inference active"
    )


    st.divider()


    # ========================================================
    # TOOLS
    # ========================================================

    st.caption(
        "AGENT TOOLS"
    )


    st.write(
        "🌐 Web search"
    )


    st.write(
        "🧮 Calculator"
    )


    st.write(
        "📚 Hybrid knowledge search"
    )


    st.caption(
        "Semantic + BM25 + RRF"
    )


    st.divider()


    # ========================================================
    # KNOWLEDGE BASE
    # ========================================================

    st.caption(
        "KNOWLEDGE BASE"
    )


    uploaded_file = (
        st.file_uploader(

            "Upload PDF",

            type=[
                "pdf"
            ],

            label_visibility=
                "collapsed"
        )
    )


    # ========================================================
    # PROCESS PDF
    # ========================================================

    if uploaded_file is not None:

        current_document = (
            st.session_state
            .document
        )


        should_process = (

            current_document
            is None

            or

            current_document.get(
                "name"
            )
            !=
            uploaded_file.name
        )


        if should_process:

            temp_path = None


            try:

                with st.status(
                    "Building knowledge base...",
                    expanded=True
                ) as status:


                    st.write(
                        "📄 Reading PDF"
                    )


                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(
                            uploaded_file
                            .getvalue()
                        )


                        temp_path = (
                            temp_file.name
                        )


                    st.write(
                        "✂️ Creating chunks"
                    )


                    st.write(
                        "🧠 Generating embeddings"
                    )


                    st.write(
                        "💾 Building vector index"
                    )


                    result = process_pdf(

                        temp_path,

                        uploaded_file.name
                    )


                    if result.get(
                        "success"
                    ):

                        st.session_state.document = {

                            "name":
                                uploaded_file.name,

                            "pages":
                                result[
                                    "pages"
                                ],

                            "chunks":
                                result[
                                    "chunks"
                                ]
                        }


                        set_active_document(
                            uploaded_file.name
                        )


                        status.update(

                            label=
                                "Knowledge base ready",

                            state=
                                "complete",

                            expanded=
                                False
                        )


                        st.rerun()


                    else:

                        status.update(

                            label=
                                "PDF processing failed",

                            state=
                                "error"
                        )


                        st.error(
                            result.get(
                                "error",
                                "Unable to process PDF."
                            )
                        )


            except Exception as error:

                st.error(
                    f"PDF error: {error}"
                )


            finally:

                if (
                    temp_path

                    and

                    os.path.exists(
                        temp_path
                    )
                ):

                    try:

                        os.remove(
                            temp_path
                        )

                    except OSError:

                        pass


    # ========================================================
    # ACTIVE DOCUMENT
    # ========================================================

    if st.session_state.document:

        document = (
            st.session_state
            .document
        )


        st.success(
            "✓ Knowledge base ready"
        )


        st.markdown(
            f"📄 **{document['name']}**"
        )


        st.caption(
            f"{document['pages']} pages · "
            f"{document['chunks']} chunks"
        )


        st.caption(
            "🧠 Semantic · 🔤 BM25 · 🔀 RRF"
        )


        if st.button(
            "Remove PDF",
            use_container_width=True
        ):

            st.session_state.document = None

            clear_active_document()

            st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title(
    "Anil AI"
)


st.caption(
    "Agentic AI · Groq · Tools · Hybrid RAG"
)


# ============================================================
# ACTIVE DOCUMENT INDICATOR
# ============================================================

if st.session_state.document:

    document = (
        st.session_state
        .document
    )


    st.caption(
        f"📚 Active knowledge base: "
        f"{document['name']}"
    )


st.divider()


# ============================================================
# EXTRACT KNOWLEDGE SOURCES
# ============================================================

def extract_kb_sources(
    tools_used
):

    sources = []


    for tool in tools_used:

        if (
            tool.get("tool")
            !=
            "knowledge_base_search"
        ):

            continue


        result = tool.get(
            "result",
            {}
        )


        if not isinstance(
            result,
            dict
        ):

            continue


        sources.extend(
            result.get(
                "sources",
                []
            )
        )


    return sources


# ============================================================
# DISPLAY TOOL EVENT
# ============================================================

def display_tool_event(
    tool
):

    tool_name = (
        tool.get(
            "tool",
            "unknown"
        )
    )


    labels = {

        "calculator":
            "🧮 Calculator",

        "web_search":
            "🌐 Web search",

        "knowledge_base_search":
            "📚 Hybrid knowledge search"
    }


    label = labels.get(
        tool_name,
        tool_name
    )


    with st.expander(
        f"{label} · completed"
    ):

        arguments = tool.get(
            "arguments",
            {}
        )


        if arguments:

            st.caption(
                "Query / Input"
            )


            st.json(
                arguments
            )


        # Do not dump huge RAG JSON
        if (
            tool_name
            !=
            "knowledge_base_search"
        ):

            result = tool.get(
                "result",
                {}
            )


            st.caption(
                "Result"
            )


            if isinstance(
                result,
                (dict, list)
            ):

                st.json(
                    result
                )

            else:

                st.write(
                    result
                )


# ============================================================
# DISPLAY RAG SOURCES
# ============================================================

def display_sources(
    sources
):

    if not sources:

        return


    with st.expander(
        f"📚 Sources · "
        f"{len(sources)} retrieved",
        expanded=False
    ):

        for index, source in enumerate(
            sources,
            start=1
        ):

            page = source.get(
                "page",
                "?"
            )


            relevance = source.get(
                "relevance",
                0
            )


            semantic = source.get(
                "semantic_relevance",
                0
            )


            keyword = source.get(
                "keyword_score",
                0
            )


            match_type = source.get(
                "match_type",
                "Hybrid"
            )


            # ================================================
            # SOURCE HEADER
            # ================================================

            st.markdown(
                f"### Source {index}"
            )


            st.markdown(
                f"**Page {page}** · "
                f"{match_type}"
            )


            st.caption(
                f"Hybrid relevance: "
                f"{relevance}% · "
                f"Semantic: "
                f"{semantic}% · "
                f"BM25: "
                f"{keyword}"
            )


            # ================================================
            # SOURCE TEXT
            # ================================================

            st.write(
                source.get(
                    "text",
                    ""
                )
            )


            if index < len(
                sources
            ):

                st.divider()


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.messages:

    st.write("")


    st.subheader(
        "How can I help?"
    )


    st.caption(
        "Ask anything, search the web, "
        "calculate, or chat with your PDF."
    )


    col1, col2 = (
        st.columns(2)
    )


    # ========================================================
    # LEFT
    # ========================================================

    with col1:

        if st.button(
            "📚 Explain RAG from my PDF",
            use_container_width=True
        ):

            st.session_state.suggested_prompt = (
                "According to my uploaded PDF, "
                "explain RAG."
            )


            st.rerun()


        if st.button(
            "🌐 Latest AI news",
            use_container_width=True
        ):

            st.session_state.suggested_prompt = (
                "What is the latest AI news today?"
            )


            st.rerun()


    # ========================================================
    # RIGHT
    # ========================================================

    with col2:

        if st.button(
            "🧠 Explain Agentic AI",
            use_container_width=True
        ):

            st.session_state.suggested_prompt = (
                "Explain Agentic AI with "
                "a practical example."
            )


            st.rerun()


        if st.button(
            "🧮 Calculate 9382 × 72",
            use_container_width=True
        ):

            st.session_state.suggested_prompt = (
                "Calculate 9382 * 72"
            )


            st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in (
    st.session_state.messages
):

    role = message.get(
        "role"
    )


    content = message.get(
        "content",
        ""
    )


    if role not in [
        "user",
        "assistant"
    ]:

        continue


    if not content:

        continue


    avatar = (

        "👤"

        if role == "user"

        else "🤖"
    )


    with st.chat_message(
        role,
        avatar=avatar
    ):

        tools_used = (
            message.get(
                "tools",
                []
            )
        )


        # ====================================================
        # TOOL TRACE
        # ====================================================

        for tool in tools_used:

            display_tool_event(
                tool
            )


        # ====================================================
        # RAG SOURCES
        # ====================================================

        sources = extract_kb_sources(
            tools_used
        )


        display_sources(
            sources
        )


        # ====================================================
        # ANSWER
        # ====================================================

        st.markdown(
            content
        )


# ============================================================
# CHAT INPUT
# ============================================================

typed_prompt = st.chat_input(
    "Message Anil AI..."
)


prompt = (

    st.session_state
    .suggested_prompt

    or

    typed_prompt
)


if (
    st.session_state
    .suggested_prompt
):

    st.session_state.suggested_prompt = None


# ============================================================
# PROCESS USER REQUEST
# ============================================================

if prompt:

    # ========================================================
    # USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role":
                "user",

            "content":
                prompt
        }
    )


    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.markdown(
            prompt
        )


    # ========================================================
    # ASSISTANT
    # ========================================================

    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        response = ""

        tools_used = []


        try:

            # =================================================
            # ACTIVE DOCUMENT
            # =================================================

            active_document = None


            if st.session_state.document:

                active_document = (
                    st.session_state
                    .document["name"]
                )


            # =================================================
            # RUN AGENT
            # =================================================

            with st.status(
                "Thinking...",
                expanded=False
            ) as agent_status:


                result = ask_llm(

                    st.session_state.messages,

                    active_document=
                        active_document
                )


                response = (
                    result.get(
                        "content",
                        ""
                    )
                )


                tools_used = (
                    result.get(
                        "tools",
                        []
                    )
                )


                agent_status.update(

                    label=
                        "Done",

                    state=
                        "complete",

                    expanded=
                        False
                )


            # =================================================
            # TOOL EVENTS
            # =================================================

            for tool in tools_used:

                display_tool_event(
                    tool
                )


            # =================================================
            # SOURCES
            # =================================================

            sources = extract_kb_sources(
                tools_used
            )


            display_sources(
                sources
            )


            # =================================================
            # FALLBACK
            # =================================================

            if not response:

                response = (
                    "I completed the request, "
                    "but no response was generated."
                )


            # =================================================
            # ANSWER
            # =================================================

            st.markdown(
                response
            )


        except Exception as error:

            response = (
                "Something went wrong while "
                "processing your request."
            )


            tools_used = []


            st.error(
                response
            )


            with st.expander(
                "Developer details"
            ):

                st.exception(
                    error
                )


    # ========================================================
    # SAVE ASSISTANT MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role":
                "assistant",

            "content":
                response,

            "tools":
                tools_used
        }
    )