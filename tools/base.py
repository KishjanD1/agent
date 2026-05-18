import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger("helpdesk_assistant.tools")

class BaseTool(ABC):
    """
    Abstract Base Class for all Tools in the Helpdesk Assistant.
    Ensures structured output, request logging, and unified error handling.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool (must be unique)."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Detailed description explaining when and how to use the tool."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema metadata describing the tool's parameters."""
        pass

    @abstractmethod
    def _execute(self, **kwargs) -> Any:
        """Core execution logic to be overridden by child classes."""
        pass

    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Wrapper that handles logging, performance timing, and error handling.
        Returns a structured dictionary response.
        """
        start_time = time.time()
        logger.info(f"Executing tool '{self.name}' with arguments: {kwargs}")
        
        try:
            result = self._execute(**kwargs)
            duration = time.time() - start_time
            logger.info(f"Tool '{self.name}' completed in {duration:.4f}s. Result: {result}")
            return {
                "status": "success",
                "tool": self.name,
                "result": result,
                "latency": duration,
                "error": None
            }
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Tool '{self.name}' failed after {duration:.4f}s. Error: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "tool": self.name,
                "result": None,
                "latency": duration,
                "error": str(e)
            }
