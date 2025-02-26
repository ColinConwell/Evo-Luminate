import re


def extractCode(source: str) -> str:
    """
    Extracts code blocks enclosed in triple backticks from a source string.

    Args:
        source (str): The source string containing code blocks

    Returns:
        str: The extracted code without the backticks
    """
    # Regular expression to match code blocks enclosed in triple backticks
    # This pattern matches:
    # - Opening triple backticks, optionally followed by a language identifier
    # - Any content (including newlines) until closing triple backticks
    pattern = r"```(?:\w*\n|\n)(.*?)```"

    # Use re.DOTALL to make '.' match newlines as well
    matches = re.findall(pattern, source, re.DOTALL)

    if not matches:
        return source

    # Return the first match
    # Note: If you want to extract all code blocks, you could return matches instead
    return matches[0]
