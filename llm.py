import json
import os

from dotenv import load_dotenv
from groq import Groq

# pyrefly: ignore [missing-import]
from tools.registry import execute_tool


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)


if not GROQ_API_KEY:

    raise ValueError(
        "GROQ_API_KEY was not found. "
        "Add GROQ_API_KEY to your .env file."
    )


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "qwen/qwen3.8-27b"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Anil AI, a helpful Agentic AI assistant.

You have three tools:

1. calculator
2. web_search
3. knowledge_base_search


============================================================
GENERAL BEHAVIOR
============================================================

Answer clearly and directly.

Do not call tools when they are unnecessary.

Never claim that a tool was used unless it was actually
called.

Use conversation history for follow-up questions.

Do not expose internal reasoning or hidden chain-of-thought.


============================================================
CALCULATOR
============================================================

Use calculator when accurate arithmetic is required.

Examples:

"Calculate 937 * 47"

"What is 900 divided by 13?"

"Add 52 and 89"


============================================================
WEB SEARCH
============================================================

Use web_search when information needs to be current.

Examples include:

latest
current
today
recent
breaking
live
this week
new announcement
current company information

Do not claim pretrained knowledge is current when a question
clearly requires recent information.

Summarize the useful information returned by web search.


============================================================
PDF KNOWLEDGE BASE
============================================================

Use knowledge_base_search when the user explicitly refers to:

uploaded PDF
uploaded document
my PDF
my document
knowledge base
according to the PDF
according to the document
from the uploaded PDF

Do not automatically search the PDF merely because a PDF
happens to be uploaded.

Example:

"Who is Virat Kohli?"

Normally answer without PDF retrieval.

Example:

"According to my uploaded PDF, explain RAG."

Use knowledge_base_search.


============================================================
HYBRID RETRIEVAL
============================================================

The knowledge base uses Hybrid Search.

It combines:

1. Semantic vector search
2. BM25 keyword search
3. Reciprocal Rank Fusion

Semantic retrieval finds information with similar meaning.

BM25 finds strong exact keyword and terminology matches.

Reciprocal Rank Fusion combines both rankings.

The final retrieval returns the top 3 chunks.


============================================================
RAG ANSWERING RULES
============================================================

When knowledge_base_search returns sources:

1. Read all retrieved sources before answering.

2. Answer the user's actual question directly.

3. Synthesize the information into one coherent answer.

4. Do not simply copy or list retrieved chunks.

5. Combine overlapping evidence.

6. Keep the explanation concise unless the user asks for
   detail.

7. Ground factual claims in the retrieved document.

8. Do not invent information unsupported by the sources.

9. Mention page numbers naturally when useful.

10. Handle obvious spelling mistakes using context.

For example:

"reag"
"ragg"
"rag"

may refer to RAG if the document evidence clearly discusses
Retrieval-Augmented Generation.

11. Do not reject a question merely because the user's term
    contains a small typo.

12. If the retrieved evidence genuinely does not contain
    enough information, say:

"I couldn't find enough relevant information in the uploaded
document."

13. Do not describe semantic search, BM25, chunks, embeddings
    or RRF in the final answer unless the user asks how the
    retrieval system works.


============================================================
RELEVANCE SCORES
============================================================

Sources may contain:

relevance
semantic_relevance
keyword_score
match_type

These are retrieval-ranking signals.

They are NOT probabilities that the final answer is correct.

Do not describe them as answer-confidence probabilities.


============================================================
TOOL HISTORY
============================================================

Previous assistant messages may contain:

[TOOLS USED FOR THIS RESPONSE]

Treat this as the authoritative record of tools used in
previous turns.

If the user asks:

"Which tool did you use?"

Answer based only on that history.


============================================================
MULTI-STEP AGENT
============================================================

You may use multiple tool calls.

For example:

User:
"Search for the current price of something and calculate
the cost of 10."

Possible workflow:

web_search
→ calculator
→ final answer

Stop calling tools once enough information has been gathered
to answer the user.
"""


# ============================================================
# TOOL DEFINITIONS
# ============================================================

TOOL_DEFINITIONS = [

    # ========================================================
    # CALCULATOR
    # ========================================================

    {
        "type": "function",

        "function": {

            "name":
                "calculator",

            "description":
                (
                    "Perform accurate arithmetic "
                    "calculations."
                ),

            "parameters": {

                "type": "object",

                "properties": {

                    "a": {

                        "type":
                            "number",

                        "description":
                            "First number."
                    },

                    "b": {

                        "type":
                            "number",

                        "description":
                            "Second number."
                    },

                    "operation": {

                        "type":
                            "string",

                        "enum": [
                            "add",
                            "subtract",
                            "multiply",
                            "divide"
                        ],

                        "description":
                            (
                                "Arithmetic operation "
                                "to perform."
                            )
                    }
                },

                "required": [
                    "a",
                    "b",
                    "operation"
                ]
            }
        }
    },


    # ========================================================
    # WEB SEARCH
    # ========================================================

    {
        "type": "function",

        "function": {

            "name":
                "web_search",

            "description":
                (
                    "Search the internet for "
                    "current, recent, latest or "
                    "time-sensitive information."
                ),

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {

                        "type":
                            "string",

                        "description":
                            (
                                "Concise web "
                                "search query."
                            )
                    },

                    "max_results": {

                        "type":
                            "integer",

                        "minimum":
                            1,

                        "maximum":
                            5,

                        "description":
                            (
                                "Number of search "
                                "results to retrieve."
                            )
                    }
                },

                "required": [
                    "query"
                ]
            }
        }
    },


    # ========================================================
    # HYBRID KNOWLEDGE SEARCH
    # ========================================================

    {
        "type": "function",

        "function": {

            "name":
                "knowledge_base_search",

            "description":
                (
                    "Search the currently uploaded "
                    "PDF knowledge base using hybrid "
                    "semantic vector search and BM25 "
                    "keyword search."
                ),

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {

                        "type":
                            "string",

                        "description":
                            (
                                "Concise search query "
                                "for retrieving relevant "
                                "information from the PDF."
                            )
                    }
                },

                "required": [
                    "query"
                ]
            }
        }
    }
]


# ============================================================
# BUILD MODEL HISTORY
# ============================================================

def build_messages(
    chat_history: list,
    active_document=None
):

    messages = [
        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT
        }
    ]


    # ========================================================
    # DOCUMENT STATE
    # ========================================================

    if active_document:

        messages.append(
            {
                "role":
                    "system",

                "content":
                    (
                        "A PDF knowledge base is "
                        "currently available.\n\n"
                        f"Active document: "
                        f"{active_document}\n\n"
                        "Do not automatically search it. "
                        "Use knowledge_base_search only "
                        "when the user's request requires "
                        "information from the document."
                    )
            }
        )

    else:

        messages.append(
            {
                "role":
                    "system",

                "content":
                    (
                        "There is currently no "
                        "active uploaded PDF."
                    )
            }
        )


    # ========================================================
    # CONVERSATION
    # ========================================================

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


        if not content:

            continue


        # ====================================================
        # USER
        # ====================================================

        if role == "user":

            messages.append(
                {
                    "role":
                        "user",

                    "content":
                        content
                }
            )

            continue


        # ====================================================
        # ASSISTANT
        # ====================================================

        tools_used = message.get(
            "tools",
            []
        )


        tool_history = ""


        if tools_used:

            lines = []


            for tool in tools_used:

                tool_name = (
                    tool.get(
                        "tool",
                        "unknown"
                    )
                )


                arguments = (
                    tool.get(
                        "arguments",
                        {}
                    )
                )


                lines.append(
                    f"- Tool: {tool_name}\n"
                    f"  Arguments: {arguments}"
                )


            tool_history = (
                "\n\n"
                "[TOOLS USED FOR THIS RESPONSE]\n"
                + "\n".join(
                    lines
                )
            )


        messages.append(
            {
                "role":
                    "assistant",

                "content":
                    content
                    + tool_history
            }
        )


    return messages


# ============================================================
# AGENT LOOP
# ============================================================

def ask_llm(
    chat_history: list,
    active_document=None
):

    messages = build_messages(

        chat_history,

        active_document
    )


    tool_events = []


    # Protect against infinite loops
    MAX_STEPS = 5


    for step in range(
        MAX_STEPS
    ):

        print(
            "\n===================================="
        )

        print(
            f"🤖 AGENT STEP: {step + 1}"
        )

        print(
            "===================================="
        )


        # ====================================================
        # MODEL
        # ====================================================

        response = (
            client
            .chat
            .completions
            .create(

                model=
                    MODEL_NAME,

                messages=
                    messages,

                tools=
                    TOOL_DEFINITIONS,

                tool_choice=
                    "auto",

                temperature=
                    0.2
            )
        )


        assistant_message = (
            response
            .choices[0]
            .message
        )


        # ====================================================
        # FINAL ANSWER
        # ====================================================

        if not assistant_message.tool_calls:

            print(
                "✅ Agent finished."
            )


            return {
                "content":
                    assistant_message.content
                    or "",

                "tools":
                    tool_events
            }


        # ====================================================
        # APPEND TOOL REQUEST
        # ====================================================

        messages.append(
            assistant_message.model_dump(
                exclude_none=True
            )
        )


        # ====================================================
        # EXECUTE REQUESTED TOOLS
        # ====================================================

        for tool_call in (
            assistant_message.tool_calls
        ):

            tool_name = (
                tool_call
                .function
                .name
            )


            # ================================================
            # ARGUMENTS
            # ================================================

            try:

                arguments = json.loads(
                    tool_call
                    .function
                    .arguments
                )

            except (
                json.JSONDecodeError,
                TypeError
            ):

                arguments = {}


            print(
                f"🔧 TOOL CALLED: "
                f"{tool_name}"
            )


            print(
                f"📦 ARGUMENTS: "
                f"{arguments}"
            )


            # ================================================
            # EXECUTE
            # ================================================

            result = execute_tool(

                tool_name,

                arguments
            )


            print(
                "✅ TOOL RESULT RECEIVED"
            )


            # ================================================
            # SAVE EVENT FOR STREAMLIT
            # ================================================

            tool_events.append(
                {
                    "tool":
                        tool_name,

                    "arguments":
                        arguments,

                    "result":
                        result
                }
            )


            # ================================================
            # RETURN RESULT TO LLM
            # ================================================

            messages.append(
                {
                    "role":
                        "tool",

                    "tool_call_id":
                        tool_call.id,

                    "content":
                        json.dumps(
                            result,
                            default=str
                        )
                }
            )


    # ========================================================
    # SAFETY FALLBACK
    # ========================================================

    return {
        "content":
            (
                "I reached the maximum number "
                "of agent steps before completing "
                "the request."
            ),

        "tools":
            tool_events
    }