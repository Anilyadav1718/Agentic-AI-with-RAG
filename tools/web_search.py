from ddgs import DDGS


def web_search(
    query: str,
    max_results: int = 5
):

    if not query or not query.strip():

        return {
            "success": False,
            "error": "Search query cannot be empty."
        }


    max_results = min(
        max(max_results, 1),
        5
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
                    "title":
                        item.get(
                            "title",
                            ""
                        ),

                    "url":
                        item.get(
                            "href",
                            ""
                        ),

                    "snippet":
                        item.get(
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