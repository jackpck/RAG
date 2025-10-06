import asyncio
from functools import wraps

def sync(func):
    """
    Decorator to turn an async function into a sync function using asyncio.run.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper