from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.logging import RequestLoggingMiddleware
from api.endpoints import router as api_router

app = FastAPI(
    title="IT Helpdesk and Knowledge Assistant API",
    description=(
        "A highly robust, manual LLM-routed Internal IT Helpdesk agent and RAG search system. "
        "Built completely from scratch without external agent frameworks."
    ),
    version="1.0.0"
)

# 1. Enable robust request/response logging middleware
app.add_middleware(RequestLoggingMiddleware)

# 2. Configure CORS middleware for painless cross-origin frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Mount helpdesk routers directly at root to match specified paths (/ingest, /query, /eval)
app.add_middleware(RequestLoggingMiddleware) # Double check to ensure middleware is active
app.include_router(api_router)

@app.get("/")
def read_root():
    """Serve the premium, interactive glassmorphic Web Chat Interface."""
    from fastapi.responses import FileResponse
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)
