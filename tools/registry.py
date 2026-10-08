from tools.web_search import web_search
from rag.document_rag import (
    knowledge_base_search,
)


TOOLS = {
    "web_search": web_search,
    "knowledge_base_search": (
        knowledge_base_search
    ),
}


def execute_tool(
    tool_name: str,
    arguments: dict,
):
    tool = TOOLS.get(tool_name)

    if tool is None:
        return {
            "success": False,
            "error": (
                f"Unknown tool: {tool_name}"
            ),
        }

    try:
        return tool(**arguments)

    except TypeError as error:
        return {
            "success": False,
            "error": (
                f"Invalid arguments: {error}"
            ),
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error),
        }