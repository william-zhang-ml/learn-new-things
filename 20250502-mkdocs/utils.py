"""Example function to document. """


def shout(some_str: str) -> str:
    """Print a string in upper-case.

    Args:
        some_str (str): string to shout

    Returns:
        str: string printed
    """
    some_str = some_str.upper()
    print(some_str)
    return some_str
