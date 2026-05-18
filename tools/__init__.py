from typing import Any, Dict
from tools.base import BaseTool
from tools.calculator import CalculatorTool
from tools.file_search import FileSearchTool

# Instantiate tools as singletons
_calculator = CalculatorTool()
_file_search = FileSearchTool()

# Central registry of available tools
TOOL_REGISTRY: Dict[str, BaseTool] = {
    _calculator.name: _calculator,
    _file_search.name: _file_search
}

def get_tool_definitions() -> list[dict]:
    """Return JSON schemas of all registered tools for LLM prompts."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters
        }
        for tool in TOOL_REGISTRY.values()
    ]

def execute_tool(name: str, arguments: dict) -> dict[str, Any]:
    """
    Look up a tool in the registry and safely run it with standard logging and timing.
    Returns a unified structured response dictionary.
    """
    if name not in TOOL_REGISTRY:
        return {
            "status": "error",
            "tool": name,
            "result": None,
            "latency": 0.0,
            "error": f"Tool '{name}' is not registered in the system."
        }
    
    return TOOL_REGISTRY[name].run(**arguments)
