import os
import tempfile

import streamlit as st

from database import (
    init_database,
    create_conversation,
    get_conversation,
    get_conversations,
    get_messages,
    save_message,
    update_conversation_title,
    delete_conversation,
    clear_all_conversations,
    delete_empty_conversations,
    generate_title,
)

from llm import (
    ask_assistant,
    ask_rag,
    MODEL_NAME,
    MAX_AGENT_STEPS,
    MAX_OUTPUT_TOKENS,
)

from rag.document_rag import (
    process_pdf,
    knowledge_base_search,
    get_active_document,
    clear_active_document,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Nexor AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DATABASE
# ============================================================

# SQLite works silently in the background.
init_database()

# Clean old empty "New chat" rows created by older versions.
delete_empty_conversations()


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "conversation_id": None,
    "mode": "assistant",
    "processed_pdf": None,
    "document": None,
    "chat_search": "",
    "confirm_clear": False,
}


for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CONVERSATION HELPERS
# ============================================================

def start_new_chat():
    """
    Start a new unsaved chat.

    IMPORTANT:
    Clicking New Chat does NOT create a SQLite row.

    A conversation is saved only after the user
    sends the first message.
    """

    st.session_state.conversation_id = None
    st.session_state.confirm_clear = False


def ensure_saved_conversation():
    """
    Create a SQLite conversation only when the user
    actually sends the first message.
    """

    conversation_id = st.session_state.conversation_id

    if conversation_id:
        conversation = get_conversation(
            conversation_id
        )

        if conversation:
            return conversation_id

    conversation_id = create_conversation(
        mode=st.session_state.mode,
        title="New chat",
    )

    st.session_state.conversation_id = (
        conversation_id
    )

    return conversation_id


def load_conversation(conversation_id):
    """
    Load a conversation from SQLite.
    """

    conversation = get_conversation(
        conversation_id
    )

    if not conversation:
        return

    st.session_state.conversation_id = (
        conversation_id
    )

    st.session_state.mode = conversation.get(
        "mode",
        "assistant",
    )

    st.session_state.confirm_clear = False


def change_mode(mode):
    """
    Change between AI Assistant and Document RAG.

    Existing conversations are preserved.
    """

    if mode not in {
        "assistant",
        "rag",
    }:
        return

    if st.session_state.mode == mode:
        return

    st.session_state.mode = mode

    # Start a fresh unsaved chat in the selected mode.
    st.session_state.conversation_id = None

    st.session_state.confirm_clear = False


# ============================================================
# AGENT ACTIVITY
# ============================================================

def display_agent_activity(
    tools,
    metrics,
):
    """
    Display AI Assistant agent observability.
    """

    if not tools and not metrics:
        return

    with st.expander(
        "🛠 Agent Activity",
        expanded=False,
    ):

        steps = metrics.get(
            "steps_used",
            1,
        )

        maximum = metrics.get(
            "max_steps",
            MAX_AGENT_STEPS,
        )

        tools_used = metrics.get(
            "tools_used",
            len(tools),
        )

        resources = metrics.get(
            "resources_used",
            0,
        )

        evidence = metrics.get(
            "evidence_score"
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "Agent Steps",
            f"{steps}/{maximum}",
        )

        col2.metric(
            "Tools Used",
            tools_used,
        )

        col3.metric(
            "Resources",
            resources,
        )

        col4.metric(
            "Evidence Score",
            (
                f"{evidence}%"
                if evidence is not None
                else "—"
            ),
        )

        st.caption(
            "Evidence Score is a retrieval heuristic, "
            "not AI confidence."
        )

        # ----------------------------------------------------
        # TOOL EVENTS
        # ----------------------------------------------------

        for event in tools:

            st.divider()

            name = event.get(
                "name",
                "tool",
            )

            status = event.get(
                "status",
                "completed",
            )

            icon = (
                "✅"
                if status == "completed"
                else "❌"
            )

            if name == "web_search":
                display_name = "Web Search"
            else:
                display_name = (
                    name
                    .replace("_", " ")
                    .title()
                )

            st.markdown(
                f"{icon} **{display_name}** · {status}"
            )

            arguments = event.get(
                "arguments",
                {},
            )

            query = arguments.get(
                "query"
            )

            if query:
                st.caption(
                    f'Search: "{query}"'
                )

            # ------------------------------------------------
            # WEB SEARCH RESOURCES
            # ------------------------------------------------

            if name == "web_search":

                result = event.get(
                    "result",
                    {},
                )

                web_results = (
                    result.get(
                        "results",
                        [],
                    )
                    or []
                )

                if web_results:
                    st.markdown(
                        "**Resources used**"
                    )

                for index, resource in enumerate(
                    web_results,
                    start=1,
                ):

                    title = (
                        resource.get("title")
                        or f"Resource {index}"
                    )

                    url = resource.get(
                        "url",
                        "",
                    )

                    snippet = resource.get(
                        "snippet",
                        "",
                    )

                    if url:
                        st.markdown(
                            f"{index}. [{title}]({url})"
                        )
                    else:
                        st.markdown(
                            f"{index}. {title}"
                        )

                    if snippet:
                        st.caption(
                            snippet[:250]
                        )


# ============================================================
# RAG ACTIVITY
# ============================================================

def display_rag_activity(
    sources,
    metrics,
):
    """
    Display Hybrid RAG retrieval information.
    """

    if not sources and not metrics:
        return

    with st.expander(
        "📚 RAG Activity",
        expanded=False,
    ):

        chunks = metrics.get(
            "chunks_used",
            len(sources),
        )

        pages = metrics.get(
            "pages_used",
            [],
        )

        evidence = metrics.get(
            "evidence_score"
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "Retrieval",
            "Hybrid",
        )

        col2.metric(
            "Chunks Used",
            chunks,
        )

        col3.metric(
            "Pages",
            len(pages),
        )

        col4.metric(
            "Evidence Score",
            (
                f"{evidence}%"
                if evidence is not None
                else "—"
            ),
        )

        st.caption(
            "Semantic Search + BM25 + "
            "Reciprocal Rank Fusion"
        )

        if pages:
            st.caption(
                "Pages used: "
                + ", ".join(
                    str(page)
                    for page in pages
                )
            )

        st.caption(
            "Evidence Score measures retrieval quality. "
            "It is not AI confidence."
        )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        for index, source in enumerate(
            sources,
            start=1,
        ):

            st.divider()

            page = source.get(
                "page",
                "?",
            )

            match_type = source.get(
                "match_type",
                "Hybrid",
            )

            relevance = source.get(
                "relevance",
                0,
            )

            semantic = source.get(
                "semantic_relevance",
                0,
            )

            keyword = source.get(
                "keyword_score",
                0,
            )

            st.markdown(
                f"**Source {index} · Page {page}**"
            )

            st.caption(
                f"{match_type}"
                f" · Hybrid {relevance}%"
                f" · Semantic {semantic}%"
                f" · BM25 {keyword}"
            )

            st.write(
                source.get(
                    "text",
                    "",
                )
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "⚡ Nexor AI"
    )

    st.caption(
        "Agentic AI + Hybrid RAG"
    )

    # ========================================================
    # NEW CHAT
    # ========================================================

    if st.button(
        "✎ New chat",
        use_container_width=True,
        type="primary",
        key="new_chat_button",
    ):

        start_new_chat()

        st.rerun()

    # ========================================================
    # SEARCH
    # ========================================================

    search_query = st.text_input(
        "Search chats",
        key="chat_search",
        placeholder="Search chats...",
        label_visibility="collapsed",
    )

    # ========================================================
    # AI MODES
    # ========================================================

    st.markdown(
        "### AI Modes"
    )

    assistant_label = (
        "✓ 💬 AI Assistant"
        if st.session_state.mode == "assistant"
        else "💬 AI Assistant"
    )

    rag_label = (
        "✓ 📚 Document RAG"
        if st.session_state.mode == "rag"
        else "📚 Document RAG"
    )

    if st.button(
        assistant_label,
        use_container_width=True,
        key="assistant_mode_button",
    ):

        change_mode(
            "assistant"
        )

        st.rerun()

    if st.button(
        rag_label,
        use_container_width=True,
        key="rag_mode_button",
    ):

        change_mode(
            "rag"
        )

        st.rerun()

    # ========================================================
    # RECENTS
    # ========================================================

    st.markdown(
        "### Recents"
    )

    # Normal:
    # only latest 5 chats.
    #
    # Search:
    # search up to 50 saved chats.

    if search_query.strip():

        conversations = get_conversations(
            search_query=search_query.strip(),
            limit=50,
        )

    else:

        conversations = get_conversations(
            search_query="",
            limit=5,
        )

    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if (
        search_query.strip()
        and not conversations
    ):

        st.caption(
            "No matching chats."
        )

    elif not conversations:

        st.caption(
            "No conversations yet."
        )

    # --------------------------------------------------------
    # RECENT CHAT BUTTONS
    # --------------------------------------------------------

    for conversation_item in conversations:

        recent_id = (
            conversation_item["id"]
        )

        title = (
            conversation_item["title"]
        )

        conversation_mode = (
            conversation_item.get(
                "mode",
                "assistant",
            )
        )

        mode_icon = (
            "📚"
            if conversation_mode == "rag"
            else "💬"
        )

        is_active = (
            recent_id
            == st.session_state.conversation_id
        )

        if is_active:

            label = (
                f"› {mode_icon} {title}"
            )

        else:

            label = (
                f"{mode_icon} {title}"
            )

        if st.button(
            label,
            key=f"recent_{recent_id}",
            use_container_width=True,
        ):

            load_conversation(
                recent_id
            )

            st.rerun()

    # ========================================================
    # CURRENT CHAT OPTIONS
    # ========================================================

    current_id = (
        st.session_state.conversation_id
    )

    current_conversation = (
        get_conversation(
            current_id
        )
        if current_id
        else None
    )

    if current_conversation:

        with st.expander(
            "⋯ Chat options"
        ):

            new_title = st.text_input(
                "Conversation title",
                value=current_conversation[
                    "title"
                ],
                key=(
                    "rename_"
                    + current_id
                ),
            )

            # ------------------------------------------------
            # RENAME CURRENT CHAT
            # ------------------------------------------------

            if st.button(
                "Rename",
                use_container_width=True,
                key=(
                    "rename_button_"
                    + current_id
                ),
            ):

                if new_title.strip():

                    update_conversation_title(
                        current_id,
                        new_title.strip(),
                    )

                    st.rerun()

            # ------------------------------------------------
            # DELETE CURRENT CHAT
            # ------------------------------------------------

            if st.button(
                "🗑 Delete this chat",
                use_container_width=True,
                key=(
                    "delete_button_"
                    + current_id
                ),
            ):

                delete_conversation(
                    current_id
                )

                start_new_chat()

                st.rerun()

    # ========================================================
    # CLEAR ALL CHATS
    # ========================================================

    st.divider()

    if not st.session_state.confirm_clear:

        if st.button(
            "🗑 Clear all chats",
            use_container_width=True,
            key="clear_all_button",
        ):

            st.session_state.confirm_clear = True

            st.rerun()

    else:

        st.warning(
            "Delete all conversation history?"
        )

        confirm_col, cancel_col = (
            st.columns(2)
        )

        # ----------------------------------------------------
        # CONFIRM
        # ----------------------------------------------------

        with confirm_col:

            if st.button(
                "Yes, clear",
                use_container_width=True,
                type="primary",
                key="confirm_clear_button",
            ):

                clear_all_conversations()

                # IMPORTANT:
                # We DO NOT modify chat_search here.
                #
                # chat_search is already instantiated above
                # as a Streamlit widget.
                #
                # Changing:
                #
                # st.session_state["chat_search"] = ""
                #
                # here causes:
                #
                # StreamlitWidgetAlreadyInstantiatedError

                st.session_state[
                    "conversation_id"
                ] = None

                st.session_state[
                    "confirm_clear"
                ] = False

                st.rerun()

        # ----------------------------------------------------
        # CANCEL
        # ----------------------------------------------------

        with cancel_col:

            if st.button(
                "Cancel",
                use_container_width=True,
                key="cancel_clear_button",
            ):

                st.session_state[
                    "confirm_clear"
                ] = False

                st.rerun()

    # ========================================================
    # SYSTEM
    # ========================================================

    with st.expander(
        "⚙ System",
        expanded=False,
    ):

        st.write(
            "**GPT-OSS 120B**"
        )

        st.caption(
            MODEL_NAME
        )

        st.write(
            f"**Agent limit:** "
            f"{MAX_AGENT_STEPS} steps"
        )

        st.write(
            f"**Output ceiling:** "
            f"{MAX_OUTPUT_TOKENS} tokens"
        )

        st.write(
            "**Reasoning:** Low"
        )

        st.write(
            "**Web results:** Max 3"
        )


# ============================================================
# CURRENT CONVERSATION
# ============================================================

conversation_id = (
    st.session_state.conversation_id
)

conversation = (
    get_conversation(
        conversation_id
    )
    if conversation_id
    else None
)

messages = (
    get_messages(
        conversation_id
    )
    if conversation_id
    else []
)


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "⚡ Nexor AI"
)

if conversation:

    st.caption(
        conversation.get(
            "title",
            "New chat",
        )
    )

else:

    st.caption(
        "New chat"
    )


# ============================================================
# AI ASSISTANT MODE
# ============================================================

if st.session_state.mode == "assistant":

    st.markdown(
        "### 💬 AI Assistant"
    )

    st.caption(
        "GPT-OSS 120B with agentic web search "
        "for current information."
    )

    # ========================================================
    # EMPTY CHAT STARTER
    # ========================================================

    if not messages:

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.markdown(
                "#### 🧠 Learn"
            )

            st.caption(
                "AI, Python, ML and "
                "technical concepts."
            )

        with col2:

            st.markdown(
                "#### 🌐 Research"
            )

            st.caption(
                "Current information with "
                "web search."
            )

        with col3:

            st.markdown(
                "#### ⚙ Build"
            )

            st.caption(
                "Agents, RAG systems and "
                "AI applications."
            )

    # ========================================================
    # DISPLAY CHAT HISTORY
    # ========================================================

    for message in messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message.get(
                    "content",
                    "",
                )
            )

            if message["role"] == "assistant":

                display_agent_activity(
                    message.get(
                        "tools",
                        [],
                    ),
                    message.get(
                        "metrics",
                        {},
                    ),
                )

    # ========================================================
    # CHAT INPUT
    # ========================================================

    prompt = st.chat_input(
        "Message Nexor AI...",
        key="assistant_input",
    )

    if prompt:

        # ----------------------------------------------------
        # SAVE CONVERSATION ONLY WHEN FIRST MESSAGE IS SENT
        # ----------------------------------------------------

        conversation_id = (
            ensure_saved_conversation()
        )

        existing_messages = (
            get_messages(
                conversation_id
            )
        )

        first_user_message = (
            len(existing_messages) == 0
        )

        # ----------------------------------------------------
        # CREATE TITLE
        # ----------------------------------------------------

        if first_user_message:

            update_conversation_title(
                conversation_id,
                generate_title(
                    prompt
                ),
            )

        # ----------------------------------------------------
        # SAVE USER MESSAGE
        # ----------------------------------------------------

        save_message(
            conversation_id,
            "user",
            prompt,
        )

        # ----------------------------------------------------
        # GET HISTORY
        # ----------------------------------------------------

        current_history = (
            get_messages(
                conversation_id
            )
        )

        # ----------------------------------------------------
        # DISPLAY USER
        # ----------------------------------------------------

        with st.chat_message(
            "user"
        ):

            st.markdown(
                prompt
            )

        # ----------------------------------------------------
        # GENERATE RESPONSE
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Nexor AI is thinking..."
            ):

                result = ask_assistant(
                    current_history
                )

            answer = result.get(
                "content",
                "",
            )

            tools = result.get(
                "tools",
                [],
            )

            metrics = result.get(
                "metrics",
                {},
            )

            st.markdown(
                answer
            )

            display_agent_activity(
                tools,
                metrics,
            )

        # ----------------------------------------------------
        # SAVE ASSISTANT RESPONSE
        # ----------------------------------------------------

        save_message(
            conversation_id,
            "assistant",
            answer,
            tools=tools,
            metrics=metrics,
        )

        st.rerun()


# ============================================================
# DOCUMENT RAG MODE
# ============================================================

else:

    st.markdown(
        "### 📚 Document RAG"
    )

    st.caption(
        "Hybrid retrieval using Semantic Search + "
        "BM25 + RRF. Answers use only the uploaded PDF."
    )

    # ========================================================
    # PDF UPLOAD
    # ========================================================

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        key="rag_pdf",
    )

    if uploaded_file:

        should_process = (
            st.session_state.processed_pdf
            != uploaded_file.name
        )

        if should_process:

            temp_path = None

            with st.spinner(
                "Building Hybrid RAG index..."
            ):

                try:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf",
                    ) as temp_file:

                        temp_file.write(
                            uploaded_file.getbuffer()
                        )

                        temp_path = (
                            temp_file.name
                        )

                    result = process_pdf(
                        temp_path,
                        uploaded_file.name,
                    )

                    if result.get(
                        "success"
                    ):

                        st.session_state[
                            "document"
                        ] = uploaded_file.name

                        st.session_state[
                            "processed_pdf"
                        ] = uploaded_file.name

                        st.success(
                            "PDF indexed successfully."
                        )

                        st.caption(
                            f"{result.get('pages', 0)} "
                            f"readable pages · "
                            f"{result.get('chunks', 0)} chunks"
                        )

                    else:

                        st.error(
                            result.get(
                                "error",
                                "PDF indexing failed.",
                            )
                        )

                finally:

                    if (
                        temp_path
                        and os.path.exists(
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

    active_document = (
        get_active_document()
    )

    if active_document:

        doc_col1, doc_col2 = (
            st.columns(
                [5, 1]
            )
        )

        with doc_col1:

            st.info(
                f"📘 Active document: "
                f"{active_document}"
            )

        with doc_col2:

            if st.button(
                "Remove PDF",
                use_container_width=True,
                key="remove_pdf_button",
            ):

                clear_active_document()

                st.session_state[
                    "document"
                ] = None

                st.session_state[
                    "processed_pdf"
                ] = None

                st.rerun()

    else:

        st.info(
            "Upload a PDF to start Document RAG."
        )

    # ========================================================
    # DISPLAY RAG HISTORY
    # ========================================================

    for message in messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message.get(
                    "content",
                    "",
                )
            )

            if message["role"] == "assistant":

                display_rag_activity(
                    message.get(
                        "sources",
                        [],
                    ),
                    message.get(
                        "metrics",
                        {},
                    ),
                )

    # ========================================================
    # RAG INPUT
    # ========================================================

    if active_document:

        rag_prompt = st.chat_input(
            "Ask about the document...",
            key="rag_input",
        )

        if rag_prompt:

            # ------------------------------------------------
            # CREATE CONVERSATION ON FIRST REAL QUESTION
            # ------------------------------------------------

            conversation_id = (
                ensure_saved_conversation()
            )

            existing_messages = (
                get_messages(
                    conversation_id
                )
            )

            first_user_message = (
                len(existing_messages) == 0
            )

            # ------------------------------------------------
            # AUTO TITLE
            # ------------------------------------------------

            if first_user_message:

                update_conversation_title(
                    conversation_id,
                    generate_title(
                        rag_prompt
                    ),
                )

            # ------------------------------------------------
            # SAVE USER
            # ------------------------------------------------

            save_message(
                conversation_id,
                "user",
                rag_prompt,
            )

            # ------------------------------------------------
            # DISPLAY USER
            # ------------------------------------------------

            with st.chat_message(
                "user"
            ):

                st.markdown(
                    rag_prompt
                )

            # ------------------------------------------------
            # HYBRID RETRIEVAL
            # ------------------------------------------------

            with st.spinner(
                "Searching the document..."
            ):

                rag_result = (
                    knowledge_base_search(
                        query=rag_prompt
                    )
                )

            sources = (
                rag_result.get(
                    "sources",
                    [],
                )
                if isinstance(
                    rag_result,
                    dict,
                )
                else []
            )

            # ------------------------------------------------
            # PREVIOUS RAG HISTORY
            # ------------------------------------------------

            rag_history = (
                get_messages(
                    conversation_id
                )
            )

            # Current user question is passed separately
            # to ask_rag, so exclude it from history.
            previous_history = (
                rag_history[:-1]
            )

            # ------------------------------------------------
            # GENERATE DOCUMENT-GROUNDED RESPONSE
            # ------------------------------------------------

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Generating grounded answer..."
                ):

                    result = ask_rag(
                        question=rag_prompt,
                        rag_result=rag_result,
                        rag_history=previous_history,
                    )

                answer = result.get(
                    "content",
                    "",
                )

                metrics = result.get(
                    "metrics",
                    {},
                )

                st.markdown(
                    answer
                )

                display_rag_activity(
                    sources,
                    metrics,
                )

            # ------------------------------------------------
            # SAVE RAG RESPONSE
            # ------------------------------------------------

            save_message(
                conversation_id,
                "assistant",
                answer,
                sources=sources,
                metrics=metrics,
            )

            st.rerun()