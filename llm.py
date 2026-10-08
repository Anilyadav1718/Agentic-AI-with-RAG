import json
import os

import streamlit as st

from dotenv import load_dotenv
from groq import Groq

# pyrefly: ignore [missing-import]
from tools.registry import execute_tool
from rag.document_rag import get_active_document


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "qwen/qwen3.8-27b"

# Keep this low because the current Groq tier/model
# has a small output-tokens-per-minute limit.
MAX_OUTPUT_TOKENS = 300

# Maximum number of agent/tool steps for one request.
MAX_STEPS = 5


# ============================================================
# API KEY
# ============================================================

# Local development:
# Load variables from .env if the file exists.
load_dotenv()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)


# Streamlit Community Cloud:
# If .env/environment variable is unavailable,
# try Streamlit App Secrets.
if not GROQ_API_KEY:

    try:
        GROQ_API_KEY = st.secrets[
            "GROQ_API_KEY"
        ]

    except Exception:
        GROQ_API_KEY = None


if not GROQ_API_KEY:

    raise ValueError(
        "GROQ_API_KEY is not configured. "
        "Add it to .env for local development "
        "or Streamlit App Secrets for deployment."
    )


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# TOOL DEFINITIONS
# ============================================================

TOOL_DEFINITIONS = [

    # --------------------------------------------------------
    # CALCULATOR
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Perform basic arithmetic calculations. "
                "Use this tool when the user asks to add, "
                "subtract, multiply, or divide numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {

                    "a": {
                        "type": "number",
                        "description": "First number."
                    },

                    "b": {
                        "type": "number",
                        "description": "Second number."
                    },

                    "operation": {
                        "type": "string",
                        "enum": [
                            "add",
                            "subtract",
                            "multiply",
                            "divide"
                        ],
                        "description": (
                            "Arithmetic operation."
                        )
                    }

                },
                "required": [
                    "a",
                    "b",
                    "operation"
                ],
                "additionalProperties": False
            }
        }
    },


    # --------------------------------------------------------
    # WEB SEARCH
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the public web for current or "
                "recent information. Use this tool for "
                "questions involving latest, today, "
                "current, recent, live, news, prices, "
                "events, or information that may have "
                "changed after the model's training."
            ),
            "parameters": {
                "type": "object",
                "properties": {

                    "query": {
                        "type": "string",
                        "description": (
                            "A concise web search query."
                        )
                    },

                    "max_results": {
                        "type": "integer",
                        "description": (
                            "Number of search results. "
                            "Use 3 or fewer."
                        ),
                        "minimum": 1,
                        "maximum": 3
                    }

                },
                "required": [
                    "query"
                ],
                "additionalProperties": False
            }
        }
    },


    # --------------------------------------------------------
    # KNOWLEDGE BASE / RAG
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "knowledge_base_search",
            "description": (
                "Search the currently uploaded PDF "
                "knowledge base using hybrid retrieval. "
                "Use this tool when the user explicitly "
                "asks about their uploaded PDF, uploaded "
                "document, uploaded file, knowledge base, "
                "or asks what their document says."
            ),
            "parameters": {
                "type": "object",
                "properties": {

                    "query": {
                        "type": "string",
                        "description": (
                            "The question or search query "
                            "for the uploaded document."
                        )
                    }

                },
                "required": [
                    "query"
                ],
                "additionalProperties": False
            }
        }
    }

]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Nexora AI, an Agentic AI assistant with tool-calling
and retrieval-augmented generation capabilities.

You have access to three tools:

1. calculator
2. web_search
3. knowledge_base_search


TOOL ROUTING RULES

Use calculator when:
- The user asks for arithmetic.
- The user asks you to calculate numbers.

Use web_search when:
- The user asks for latest information.
- The user says "today".
- The user says "current".
- The user says "recent".
- The user asks for news.
- The user asks for live information.
- The answer depends on information that may have changed.

Use knowledge_base_search when:
- The user explicitly mentions their uploaded PDF.
- The user says "my PDF".
- The user says "uploaded PDF".
- The user says "uploaded document".
- The user says "uploaded file".
- The user says "knowledge base".
- The user asks "according to my document".
- The user asks what information exists inside the uploaded document.

Do NOT use knowledge_base_search merely because a PDF happens
to be uploaded.

For general knowledge questions that do not require current
information, answer directly without using a tool.


RAG RULES

When knowledge_base_search returns multiple sources:

- Read all returned sources.
- Synthesize the evidence into one answer.
- Do not simply list or copy the retrieved chunks.
- Do not reproduce entire chunks.
- Use the retrieved information as the basis for the answer.

The retrieval system may provide:
- hybrid relevance
- semantic relevance
- BM25 scores
- match type

These values are retrieval ranking signals.
They are NOT probabilities and are NOT calibrated confidence scores.

The backend controls how many RAG chunks are retrieved.
Do not attempt to choose top_k.

If the user makes an obvious typo such as:
"reag", "ragg", or similar,
and the document context clearly indicates they mean "RAG",
interpret the intended term appropriately.


WEB SEARCH RULES

When using web_search:

- Use a concise search query.
- Request at most 3 results.
- Summarize the useful information.
- Do not reproduce entire search snippets.
- Do not unnecessarily repeat the same web search.
- Prefer one search unless another search is genuinely required.


RESPONSE STYLE

Keep responses concise and useful.

For ordinary questions:
- Prefer a short direct answer.

For web-search questions:
- Summarize the important findings.
- Prefer approximately 200 words or fewer unless the user
  explicitly requests more detail.

For document questions:
- Synthesize the retrieved evidence.
- Do not repeat full retrieved chunks.

Do not claim that a tool was used unless it was actually used.
"""


# ============================================================
# BUILD CHAT HISTORY
# ============================================================

def build_messages(
    chat_history,
    active_document=None
):
    """
    Convert Streamlit chat history into Groq messages.

    Historical tool metadata is converted into plain text so
    the model can answer follow-up questions such as:
    "Which tool did you use?"

    active_document may be passed by app.py.
    If app.py does not pass it, fall back to the RAG module.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]


    # --------------------------------------------------------
    # ACTIVE DOCUMENT STATUS
    # --------------------------------------------------------

    if not active_document:

        try:
            active_document = (
                get_active_document()
            )

        except Exception:
            active_document = None


    if active_document:

        messages.append(
            {
                "role": "system",
                "content": (
                    "A PDF knowledge base is currently "
                    f"active: {active_document}. "
                    "Do not search it unless the user's "
                    "request explicitly refers to the "
                    "uploaded document or knowledge base."
                )
            }
        )

    else:

        messages.append(
            {
                "role": "system",
                "content": (
                    "There is currently no active "
                    "uploaded PDF knowledge base."
                )
            }
        )


    # --------------------------------------------------------
    # CONVERSATION HISTORY
    # --------------------------------------------------------

    for message in chat_history:

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


        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        if role == "user":

            messages.append(
                {
                    "role": "user",
                    "content": content
                }
            )

            continue


        # ----------------------------------------------------
        # ASSISTANT MESSAGE
        # ----------------------------------------------------

        assistant_content = (
            content
            if content
            else ""
        )


        # ----------------------------------------------------
        # HISTORICAL TOOL METADATA
        # ----------------------------------------------------

        tools_used = message.get(
            "tools",
            []
        )


        if tools_used:

            tool_history = [
                "",
                "[TOOLS USED FOR THIS RESPONSE]"
            ]


            for tool_event in tools_used:

                tool_name = tool_event.get(
                    "name",
                    "unknown"
                )

                arguments = tool_event.get(
                    "arguments",
                    {}
                )


                tool_history.append(
                    f"- Tool: {tool_name}"
                )

                tool_history.append(
                    "  Arguments: "
                    + json.dumps(
                        arguments,
                        ensure_ascii=False
                    )
                )


            assistant_content += (
                "\n"
                + "\n".join(
                    tool_history
                )
            )


        messages.append(
            {
                "role": "assistant",
                "content": assistant_content
            }
        )


    return messages


# ============================================================
# PARSE TOOL ARGUMENTS
# ============================================================

def parse_tool_arguments(
    raw_arguments
):
    """
    Convert Groq's JSON argument string into a Python dict.
    """

    if not raw_arguments:
        return {}


    if isinstance(
        raw_arguments,
        dict
    ):
        return raw_arguments


    try:

        parsed = json.loads(
            raw_arguments
        )


        if isinstance(
            parsed,
            dict
        ):
            return parsed


        return {}


    except Exception:

        return {}


# ============================================================
# EXTRACT RAG SOURCES
# ============================================================

def extract_sources(
    tool_name,
    tool_result
):
    """
    Extract source metadata from the RAG tool so app.py can
    display the retrieved PDF chunks.
    """

    if tool_name != "knowledge_base_search":
        return []


    if not isinstance(
        tool_result,
        dict
    ):
        return []


    if not tool_result.get(
        "success"
    ):
        return []


    return tool_result.get(
        "sources",
        []
    ) or []


# ============================================================
# ASK LLM
# ============================================================

def ask_llm(
    chat_history,
    active_document=None
):
    """
    Run the Agentic AI loop.

    Flow:

    User
      ↓
    Groq
      ↓
    Tool call?
      ├── Yes → execute tool → return result → Groq
      └── No  → final answer

    active_document is accepted so this function is compatible
    with the existing app.py.

    max_tokens is intentionally kept low because the current
    Groq tier/model has a small output-tokens-per-minute quota.
    """


    # ========================================================
    # BUILD MESSAGES
    # ========================================================

    messages = build_messages(
        chat_history,
        active_document=active_document
    )


    tool_events = []

    all_sources = []


    # ========================================================
    # AGENT LOOP
    # ========================================================

    for step in range(
        1,
        MAX_STEPS + 1
    ):

        print(
            "\n"
            "===================================="
        )

        print(
            f"🤖 AGENT STEP: {step}"
        )

        print(
            "===================================="
        )


        # ----------------------------------------------------
        # CALL GROQ
        # ----------------------------------------------------

        response = (
            client
            .chat
            .completions
            .create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=MAX_OUTPUT_TOKENS
            )
        )


        assistant_message = (
            response
            .choices[0]
            .message
        )


        tool_calls = (
            assistant_message.tool_calls
            or []
        )


        # ====================================================
        # FINAL ANSWER
        # ====================================================

        if not tool_calls:

            final_content = (
                assistant_message.content
                or ""
            )


            print(
                "✅ Agent finished."
            )


            return {
                "content": final_content,
                "tools": tool_events,
                "sources": all_sources
            }


        # ====================================================
        # TOOL CALL MESSAGE
        # ====================================================

        messages.append(
            {
                "role": "assistant",
                "content": (
                    assistant_message.content
                    or ""
                ),
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": (
                                tool_call
                                .function
                                .name
                            ),
                            "arguments": (
                                tool_call
                                .function
                                .arguments
                            )
                        }
                    }
                    for tool_call
                    in tool_calls
                ]
            }
        )


        # ====================================================
        # EXECUTE EACH TOOL
        # ====================================================

        for tool_call in tool_calls:

            tool_name = (
                tool_call
                .function
                .name
            )


            arguments = (
                parse_tool_arguments(
                    tool_call
                    .function
                    .arguments
                )
            )


            # ------------------------------------------------
            # WEB SEARCH SAFETY LIMIT
            # ------------------------------------------------

            if tool_name == "web_search":

                try:

                    requested_results = int(
                        arguments.get(
                            "max_results",
                            3
                        )
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    requested_results = 3


                arguments[
                    "max_results"
                ] = max(
                    1,
                    min(
                        requested_results,
                        3
                    )
                )


            # ------------------------------------------------
            # LOG TOOL CALL
            # ------------------------------------------------

            print(
                f"🔧 TOOL CALLED: "
                f"{tool_name}"
            )


            print(
                f"Arguments: "
                f"{arguments}"
            )


            # ------------------------------------------------
            # EXECUTE TOOL
            # ------------------------------------------------

            tool_result = execute_tool(
                tool_name,
                arguments
            )


            if isinstance(
                tool_result,
                dict
            ):

                print(
                    "Tool success: "
                    f"{tool_result.get('success')}"
                )

            else:

                print(
                    "Tool executed."
                )


            # ------------------------------------------------
            # SAVE TOOL EVENT
            # ------------------------------------------------

            tool_events.append(
                {
                    "name": tool_name,
                    "arguments": arguments,
                    "result": tool_result
                }
            )


            # ------------------------------------------------
            # SAVE RAG SOURCES
            # ------------------------------------------------

            sources = extract_sources(
                tool_name,
                tool_result
            )


            if sources:

                all_sources.extend(
                    sources
                )


            # ------------------------------------------------
            # RETURN TOOL RESULT TO GROQ
            # ------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": (
                        tool_call.id
                    ),
                    "content": json.dumps(
                        tool_result,
                        ensure_ascii=False,
                        default=str
                    )
                }
            )


    # ========================================================
    # MAX STEPS REACHED
    # ========================================================

    return {
        "content": (
            "I reached the maximum number of "
            "agent steps before completing the request."
        ),
        "tools": tool_events,
        "sources": all_sources
    }