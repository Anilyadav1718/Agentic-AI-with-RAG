from tools.calculator import calculator

from tools.web_search import (
    web_search
)

from rag.document_rag import (
    knowledge_base_search
)


# ============================================================
# TOOL REGISTRY
# ============================================================

TOOLS = {

    "calculator":
        calculator,

    "web_search":
        web_search,

    "knowledge_base_search":
        knowledge_base_search,
}


# ============================================================
# EXECUTE TOOL
# ============================================================

def execute_tool(
    tool_name: str,
    arguments: dict
):

    tool = TOOLS.get(
        tool_name
    )


    if tool is None:

        return {
            "success": False,
            "error": (
                f"Unknown tool: "
                f"{tool_name}"
            )
        }


    try:

        return tool(
            **arguments
        )


    except TypeError as error:

        return {
            "success": False,
            "error": (
                f"Invalid arguments: "
                f"{error}"
            )
        }


    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }