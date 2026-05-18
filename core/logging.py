import logging
import sys
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("helpdesk_assistant")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        path = request.url.path
        method = request.method
        
        # Log incoming request
        logger.info(f"Incoming request: {method} {path} from client {request.client.host if request.client else 'unknown'}")
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            # Log successful response
            logger.info(f"Response: {method} {path} - Status {response.status_code} - Latency: {process_time:.4f}s")
            return response
        except Exception as e:
            process_time = time.time() - start_time
            # Log failure
            logger.error(f"Request failed: {method} {path} - Error: {str(e)} - Latency: {process_time:.4f}s", exc_info=True)
            raise e
