from ddgs import DDGS


def web_search(
    query: str,
    max_results: int = 3
):
    """
    Search the web using DDGS.

    The number of results is intentionally limited
    to reduce the amount of context sent back to the LLM.
    """

    if not query or not query.strip():
        return {
            "success": False,
            "error": "Search query cannot be empty."
        }

    # Keep results small to reduce LLM token usage
    max_results = min(
        max(max_results, 1),
        3
    )

    try:
        raw_results = DDGS().text(
            query,
            max_results=max_results
        )

        results = []

        for item in raw_results:
            results.append(
                {
                    "title": item.get(
                        "title",
                        ""
                    ),
                    "url": item.get(
                        "href",
                        ""
                    ),
                    "snippet": item.get(
                        "body",
                        ""
                    )
                }
            )

        return {
            "success": True,
            "query": query,
            "results": results
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error)
        }