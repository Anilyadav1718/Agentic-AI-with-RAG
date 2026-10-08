def calculator(
    a: float,
    b: float,
    operation: str
):

    if operation == "add":
        result = a + b

    elif operation == "subtract":
        result = a - b

    elif operation == "multiply":
        result = a * b

    elif operation == "divide":

        if b == 0:

            return {
                "success": False,
                "error": "Cannot divide by zero."
            }

        result = a / b

    else:

        return {
            "success": False,
            "error": (
                f"Unsupported operation: "
                f"{operation}"
            )
        }


    return {
        "success": True,
        "a": a,
        "b": b,
        "operation": operation,
        "result": result
    }