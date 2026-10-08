import json
import os
import re
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# pyrefly: ignore [missing-import]
from tools.registry import execute_tool


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "Nexor AI"

MODEL_NAME = "openai/gpt-oss-120b"

MAX_AGENT_STEPS = 5

# Higher ceiling so technical/tutorial responses can finish.
# This is a maximum, not a guaranteed usage amount.
MAX_OUTPUT_TOKENS = 3000

MAX_WEB_RESULTS = 3

# Low reasoning leaves more room for the visible answer.
DEFAULT_REASONING_EFFORT = "low"
RAG_REASONING_EFFORT = "low"


# ============================================================
# API KEY
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = None


if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Add it to .env locally or Streamlit Secrets "
        "when deployed."
    )


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# WEB SEARCH TOOL
# IMPORTANT:
# max_results is NOT exposed to the model.
# Python controls it.
# ============================================================

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the public web for current, latest, "
            "recent, live, news, or time-sensitive information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A concise and specific web search query."
                    ),
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


ASSISTANT_TOOLS = [
    WEB_SEARCH_TOOL
]


# ============================================================
# ASSISTANT SYSTEM PROMPT
# ============================================================

ASSISTANT_SYSTEM_PROMPT = """
You are Nexor AI, an advanced Agentic AI assistant.

You can use a web_search tool.

GENERAL KNOWLEDGE

Answer stable general knowledge, programming, Python,
AI, machine learning, software engineering, RAG,
Agentic AI, career, interview, educational and
conceptual questions directly.

WEB SEARCH

Use web_search when the user asks for information that is:

- current
- latest
- today
- recent
- live
- news-related
- time-sensitive
- likely to have changed recently

Use one concise and specific search query.

Search only once when one successful search gives enough
information.

After receiving web results, synthesize those results
into the final answer.

Do not repeatedly search for slightly different versions
of the same query.

CURRENT INFORMATION SAFETY

Never describe information as current, latest, recent,
today, or "as of" the current date unless web_search
was actually executed.

Never invent recent developments.

OUTPUT FORMATTING

Use clean Markdown.

Do not output raw HTML formatting such as:

<br>
<br/>
<p>
</p>
<div>
</div>

Use normal Markdown paragraphs, headings, lists,
tables and fenced code blocks.

PROGRAMMING ANSWERS

When generating code:

- Use fenced Markdown code blocks.
- Specify the programming language.
- Generate syntactically complete code.
- Never intentionally stop in the middle of a function.
- Never intentionally stop in the middle of a code block.
- Do not use HTML line-break tags.
- Explain important code when useful.

ANSWER QUALITY

Give clear, complete and technically correct answers.

For simple questions, be concise.

For complex technical questions, provide enough detail
to properly answer the request.

Prefer a complete focused answer over unnecessary verbosity.

Never intentionally stop in the middle of:

- a sentence
- a function
- a code block
- a numbered explanation
- a Markdown section

BROAD EDUCATIONAL REQUESTS

If the user asks for an extremely broad subject such as
"teach me Python completely", "teach me AI", or another
topic that would require a very large response:

1. Give a clear learning roadmap.
2. Teach a useful and complete first section.
3. Finish all examples and code blocks.
4. Clearly identify what should be learned next.
5. Do not attempt to squeeze an entire textbook into one answer.

It is better to provide one complete lesson than several
incomplete lessons.
"""


# ============================================================
# RAG SYSTEM PROMPT
# ============================================================

RAG_SYSTEM_PROMPT = """
You are Nexor AI operating in Document RAG mode.

Answer ONLY using the supplied retrieved document context.

RULES

1. Use only the supplied document context.

2. Do not use web information.

3. Do not use general model knowledge to fill missing facts.

4. If the retrieved document context does not contain enough
   information, clearly state that the answer could not be
   found in the uploaded document.

5. Synthesize multiple retrieved chunks when useful.

6. Do not simply dump the retrieved chunks.

7. Mention document page numbers when useful.

8. Retrieval scores are ranking and evidence signals.
   They are not probabilities of correctness.

OUTPUT FORMATTING

Use clean Markdown.

Do not output raw HTML tags such as:

<br>
<p>
<div>

Use fenced Markdown code blocks for code.

Never intentionally stop in the middle of a sentence,
function, code block, list or section.

For broad questions about the document, provide a focused
summary rather than unnecessarily reproducing the entire
document.
"""


# ============================================================
# CLEAN MODEL OUTPUT
# ============================================================

def clean_model_output(content: str) -> str:
    """
    Convert common unwanted HTML formatting to clean Markdown.
    """

    if not content:
        return ""

    # <br>, <br/>, <br />
    content = re.sub(
        r"<br\s*/?>",
        "\n",
        content,
        flags=re.IGNORECASE,
    )

    # Paragraph tags
    content = re.sub(
        r"</?p[^>]*>",
        "",
        content,
        flags=re.IGNORECASE,
    )

    # Div tags
    content = re.sub(
        r"</?div[^>]*>",
        "",
        content,
        flags=re.IGNORECASE,
    )

    # Excessive blank lines
    content = re.sub(
        r"\n{3,}",
        "\n\n",
        content,
    )

    return content.strip()


# ============================================================
# PARSE TOOL ARGUMENTS
# ============================================================

def parse_tool_arguments(raw_arguments):
    if not raw_arguments:
        return {}

    if isinstance(raw_arguments, dict):
        return raw_arguments

    try:
        parsed = json.loads(raw_arguments)

        if isinstance(parsed, dict):
            return parsed

    except Exception:
        pass

    return {}


# ============================================================
# WEB EVIDENCE SCORE
# ============================================================

def calculate_web_evidence_score(tool_events):
    """
    Heuristic retrieval score.

    IMPORTANT:
    This is NOT model confidence and NOT a probability
    that the answer is correct.
    """

    web_results = []

    for event in tool_events:

        if event.get("name") != "web_search":
            continue

        result = event.get("result", {})

        if not result.get("success", False):
            continue

        web_results.extend(
            result.get("results", []) or []
        )

    if not web_results:
        return None

    usable_results = [
        result
        for result in web_results
        if result.get("title") and result.get("url")
    ]

    if not usable_results:
        return None

    domains = set()

    for result in usable_results:

        url = result.get("url", "")

        if not url:
            continue

        try:
            domain = (
                urlparse(url)
                .netloc
                .lower()
            )

            if domain.startswith("www."):
                domain = domain[4:]

            if domain:
                domains.add(domain)

        except Exception:
            pass

    # Up to 45 points for usable result count
    quantity_score = min(
        len(usable_results) / MAX_WEB_RESULTS,
        1.0,
    ) * 45

    # Up to 35 points for domain diversity
    diversity_score = min(
        len(domains) / MAX_WEB_RESULTS,
        1.0,
    ) * 35

    # Successful retrieval
    success_score = 20

    score = (
        quantity_score
        + diversity_score
        + success_score
    )

    return round(
        min(score, 100),
        1,
    )


# ============================================================
# RAG EVIDENCE SCORE
# ============================================================

def calculate_rag_evidence_score(sources):
    """
    Hybrid retrieval evidence heuristic.

    Not a probability of correctness.
    """

    if not sources:
        return None

    relevance_scores = [
        float(
            source.get(
                "relevance",
                0,
            )
        )
        for source in sources
    ]

    semantic_scores = [
        float(
            source.get(
                "semantic_relevance",
                0,
            )
        )
        for source in sources
    ]

    relevance_average = (
        sum(relevance_scores)
        / len(relevance_scores)
    )

    semantic_average = (
        sum(semantic_scores)
        / len(semantic_scores)
    )

    both_count = sum(
        1
        for source in sources
        if source.get("match_type")
        == "Semantic + Keyword"
    )

    fusion_bonus = (
        both_count
        / len(sources)
    ) * 10

    score = (
        relevance_average * 0.50
        + semantic_average * 0.40
        + fusion_bonus
    )

    return round(
        max(
            0,
            min(score, 100),
        ),
        1,
    )


# ============================================================
# BUILD CHAT HISTORY
# ============================================================

def build_assistant_messages(chat_history):
    messages = [
        {
            "role": "system",
            "content": ASSISTANT_SYSTEM_PROMPT,
        }
    ]

    # SQLite stores the complete history.
    # Only the latest messages are sent to the model.
    # This helps reduce input-token usage.
    recent_history = chat_history[-12:]

    for message in recent_history:

        role = message.get("role")

        if role not in {
            "user",
            "assistant",
        }:
            continue

        content = message.get(
            "content",
            "",
        )

        if not content:
            continue

        messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    return messages


# ============================================================
# AGENT METRICS
# ============================================================

def build_agent_metrics(
    tool_events,
    steps_used,
    truncated=False,
):
    resources_used = 0

    for event in tool_events:

        result = event.get(
            "result",
            {},
        )

        resources_used += len(
            result.get(
                "results",
                [],
            )
            or []
        )

    return {
        "steps_used": steps_used,
        "max_steps": MAX_AGENT_STEPS,
        "tools_used": len(tool_events),
        "resources_used": resources_used,
        "evidence_score": (
            calculate_web_evidence_score(
                tool_events
            )
        ),
        "truncated": truncated,
    }


# ============================================================
# RATE LIMIT CHECK
# ============================================================

def is_rate_limit_error(error_text: str) -> bool:
    lowered = error_text.lower()

    return (
        "429" in error_text
        or "rate_limit" in lowered
        or "ratelimit" in lowered
        or "rate limit" in lowered
    )


# ============================================================
# TRUNCATION MESSAGE
# ============================================================

def add_truncation_notice(
    answer: str,
    truncated: bool,
) -> str:

    if not truncated:
        return answer

    return (
        answer
        + "\n\n---\n\n"
        + "⚠️ **This response reached the output-token "
          "limit.** Send **continue** and Nexor AI can "
          "continue from this point."
    )


# ============================================================
# AI ASSISTANT
# ============================================================

def ask_assistant(chat_history):
    messages = build_assistant_messages(
        chat_history
    )

    tool_events = []

    used_tools = set()

    steps_used = 0

    for step in range(
        1,
        MAX_AGENT_STEPS + 1,
    ):

        steps_used = step

        # ====================================================
        # FIRST / NORMAL MODEL CALL
        # ====================================================

        try:
            response = (
                client.chat
                .completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=ASSISTANT_TOOLS,
                    tool_choice="auto",
                    reasoning_effort=(
                        DEFAULT_REASONING_EFFORT
                    ),
                    reasoning_format="hidden",
                    temperature=0.3,
                    max_completion_tokens=(
                        MAX_OUTPUT_TOKENS
                    ),
                )
            )

        except Exception as error:

            error_text = str(error)

            if is_rate_limit_error(
                error_text
            ):
                return {
                    "content": (
                        "Groq rate limit reached. "
                        "Please wait briefly and try again."
                    ),
                    "tools": tool_events,
                    "metrics": build_agent_metrics(
                        tool_events,
                        steps_used,
                    ),
                }

            return {
                "content": (
                    "The AI service returned an error:\n\n"
                    f"`{error_text}`"
                ),
                "tools": tool_events,
                "metrics": build_agent_metrics(
                    tool_events,
                    steps_used,
                ),
            }

        choice = response.choices[0]

        assistant_message = choice.message

        finish_reason = choice.finish_reason

        tool_calls = (
            assistant_message.tool_calls
            or []
        )

        # ====================================================
        # NO TOOL CALL = FINAL ANSWER
        # ====================================================

        if not tool_calls:

            answer = clean_model_output(
                assistant_message.content
                or ""
            )

            truncated = (
                finish_reason == "length"
            )

            answer = add_truncation_notice(
                answer,
                truncated,
            )

            return {
                "content": answer,
                "tools": tool_events,
                "metrics": build_agent_metrics(
                    tool_events,
                    steps_used,
                    truncated,
                ),
            }

        # ====================================================
        # ADD ASSISTANT TOOL CALL TO MESSAGE HISTORY
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
                            ),
                        },
                    }
                    for tool_call in tool_calls
                ],
            }
        )

        # ====================================================
        # EXECUTE TOOL CALLS
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
            # REPEATED TOOL PROTECTION
            # ------------------------------------------------

            if tool_name in used_tools:

                duplicate_result = {
                    "success": False,
                    "error": (
                        f"{tool_name} has already been used "
                        "for this request. Use the previous "
                        "tool result and answer the user."
                    ),
                }

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": (
                            tool_call.id
                        ),
                        "content": json.dumps(
                            duplicate_result,
                            ensure_ascii=False,
                        ),
                    }
                )

                continue

            # ------------------------------------------------
            # WEB RESULT LIMIT
            # ------------------------------------------------

            if tool_name == "web_search":

                # Model cannot control this.
                arguments[
                    "max_results"
                ] = MAX_WEB_RESULTS

            # ------------------------------------------------
            # EXECUTE
            # ------------------------------------------------

            tool_result = execute_tool(
                tool_name,
                arguments,
            )

            success = (
                isinstance(
                    tool_result,
                    dict,
                )
                and tool_result.get(
                    "success",
                    False,
                )
            )

            status = (
                "completed"
                if success
                else "failed"
            )

            used_tools.add(
                tool_name
            )

            tool_events.append(
                {
                    "name": tool_name,
                    "arguments": arguments,
                    "result": tool_result,
                    "status": status,
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": (
                        tool_call.id
                    ),
                    "content": json.dumps(
                        tool_result,
                        ensure_ascii=False,
                        default=str,
                    ),
                }
            )

        # ====================================================
        # WEB SEARCH COMPLETED
        #
        # Force final synthesis instead of allowing the model
        # to call web_search repeatedly.
        # ====================================================

        if "web_search" in used_tools:

            steps_used = min(
                step + 1,
                MAX_AGENT_STEPS,
            )

            try:
                final_response = (
                    client.chat
                    .completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        tool_choice="none",
                        reasoning_effort=(
                            DEFAULT_REASONING_EFFORT
                        ),
                        reasoning_format="hidden",
                        temperature=0.3,
                        max_completion_tokens=(
                            MAX_OUTPUT_TOKENS
                        ),
                    )
                )

            except Exception as error:

                error_text = str(error)

                if is_rate_limit_error(
                    error_text
                ):
                    content = (
                        "Web search completed, but Groq's "
                        "rate limit was reached while generating "
                        "the final answer. Please wait briefly "
                        "and try again."
                    )

                else:
                    content = (
                        "Web search completed, but final "
                        "answer generation failed:\n\n"
                        f"`{error_text}`"
                    )

                return {
                    "content": content,
                    "tools": tool_events,
                    "metrics": (
                        build_agent_metrics(
                            tool_events,
                            steps_used,
                        )
                    ),
                }

            final_choice = (
                final_response
                .choices[0]
            )

            answer = clean_model_output(
                final_choice
                .message
                .content
                or ""
            )

            truncated = (
                final_choice.finish_reason
                == "length"
            )

            answer = add_truncation_notice(
                answer,
                truncated,
            )

            return {
                "content": answer,
                "tools": tool_events,
                "metrics": (
                    build_agent_metrics(
                        tool_events,
                        steps_used,
                        truncated,
                    )
                ),
            }

    # ========================================================
    # MAX AGENT STEPS
    # ========================================================

    return {
        "content": (
            "I reached the maximum agent execution limit "
            "before completing the request."
        ),
        "tools": tool_events,
        "metrics": build_agent_metrics(
            tool_events,
            steps_used,
        ),
    }


# ============================================================
# DOCUMENT RAG
# ============================================================

def ask_rag(
    question,
    rag_result,
    rag_history=None,
):

    # ========================================================
    # VALIDATION
    # ========================================================

    if not rag_result:
        return {
            "content": (
                "No document retrieval result "
                "was available."
            ),
            "metrics": {
                "retrieval": (
                    "Semantic + BM25 + RRF"
                ),
                "chunks_used": 0,
                "pages_used": [],
                "evidence_score": None,
                "truncated": False,
            },
        }

    if not rag_result.get(
        "success",
        False,
    ):
        return {
            "content": rag_result.get(
                "error",
                "Document retrieval failed.",
            ),
            "metrics": {
                "retrieval": (
                    "Semantic + BM25 + RRF"
                ),
                "chunks_used": 0,
                "pages_used": [],
                "evidence_score": None,
                "truncated": False,
            },
        }

    sources = (
        rag_result.get(
            "sources",
            [],
        )
        or []
    )

    if not sources:
        return {
            "content": (
                "I couldn't find enough information in "
                "the uploaded document to answer that question."
            ),
            "metrics": {
                "retrieval": (
                    "Semantic + BM25 + RRF"
                ),
                "chunks_used": 0,
                "pages_used": [],
                "evidence_score": None,
                "truncated": False,
            },
        }

    # ========================================================
    # BUILD RETRIEVED CONTEXT
    # ========================================================

    context_parts = []

    for index, source in enumerate(
        sources,
        start=1,
    ):

        context_parts.append(
            (
                f"SOURCE {index}\n"
                f"Page: {source.get('page', '?')}\n"
                f"Match: "
                f"{source.get('match_type', 'Hybrid')}\n\n"
                f"{source.get('text', '')}"
            )
        )

    document_context = (
        "\n\n---\n\n".join(
            context_parts
        )
    )

    # ========================================================
    # MESSAGES
    # ========================================================

    messages = [
        {
            "role": "system",
            "content": RAG_SYSTEM_PROMPT,
        }
    ]

    # Only a few recent RAG messages are needed.
    if rag_history:

        for message in rag_history[-6:]:

            role = message.get("role")

            if role not in {
                "user",
                "assistant",
            }:
                continue

            content = message.get(
                "content",
                "",
            )

            if not content:
                continue

            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    messages.append(
        {
            "role": "user",
            "content": (
                "DOCUMENT CONTEXT\n"
                "================\n\n"
                f"{document_context}\n\n"
                "USER QUESTION\n"
                "=============\n\n"
                f"{question}\n\n"
                "Answer the question using ONLY the "
                "document context above."
            ),
        }
    )

    # ========================================================
    # RAG METRICS
    # ========================================================

    pages = sorted(
        {
            str(
                source.get(
                    "page",
                    "?",
                )
            )
            for source in sources
        }
    )

    evidence_score = (
        calculate_rag_evidence_score(
            sources
        )
    )

    # ========================================================
    # GENERATE DOCUMENT-GROUNDED ANSWER
    # ========================================================

    try:
        response = (
            client.chat
            .completions.create(
                model=MODEL_NAME,
                messages=messages,
                reasoning_effort=(
                    RAG_REASONING_EFFORT
                ),
                reasoning_format="hidden",
                temperature=0.2,
                max_completion_tokens=(
                    MAX_OUTPUT_TOKENS
                ),
            )
        )

    except Exception as error:

        error_text = str(error)

        if is_rate_limit_error(
            error_text
        ):
            content = (
                "Groq rate limit reached. "
                "Please wait briefly and try again."
            )

        else:
            content = (
                "The AI service returned an error:\n\n"
                f"`{error_text}`"
            )

        return {
            "content": content,
            "metrics": {
                "retrieval": (
                    "Semantic + BM25 + RRF"
                ),
                "chunks_used": len(
                    sources
                ),
                "pages_used": pages,
                "evidence_score": (
                    evidence_score
                ),
                "truncated": False,
            },
        }

    choice = response.choices[0]

    answer = clean_model_output(
        choice.message.content
        or ""
    )

    truncated = (
        choice.finish_reason
        == "length"
    )

    answer = add_truncation_notice(
        answer,
        truncated,
    )

    return {
        "content": answer,
        "metrics": {
            "retrieval": (
                "Semantic + BM25 + RRF"
            ),
            "chunks_used": len(
                sources
            ),
            "pages_used": pages,
            "evidence_score": (
                evidence_score
            ),
            "truncated": truncated,
        },
    }