from ddgs import DDGS


MAX_WEB_RESULTS = 3


def web_search(
    query: str,
    max_results: int = MAX_WEB_RESULTS,
):
    if not query or not query.strip():
        return {
            "success": False,
            "error": "Search query cannot be empty.",
            "results": [],
        }

    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = MAX_WEB_RESULTS

    max_results = max(
        1,
        min(
            max_results,
            MAX_WEB_RESULTS,
        ),
    )

    try:
        raw_results = DDGS().text(
            query.strip(),
            max_results=max_results,
        )

        results = []

        for item in raw_results:
            results.append(
                {
                    "title": item.get(
                        "title",
                        "",
                    ),
                    "url": item.get(
                        "href",
                        "",
                    ),
                    "snippet": item.get(
                        "body",
                        "",
                    ),
                }
            )

        if not results:
            return {
                "success": False,
                "query": query,
                "error": "No search results found.",
                "results": [],
            }

        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": results,
        }

    except Exception as error:
        return {
            "success": False,
            "query": query,
            "error": str(error),
            "results": [],
        }