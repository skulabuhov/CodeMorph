import json
from typing import Any, Dict, Callable

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Return the same text that was provided",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to echo"}
                },
                "required": ["text"],
            },
        },
    }
]

async def echo(text: str) -> str:
    return text

TOOL_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "echo": echo,
}
