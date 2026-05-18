# IT Helpdesk & Knowledge Assistant

An advanced, highly robust, and manual LLM-routed Internal IT Helpdesk Assistant and RAG Search System built completely from scratch without external agent frameworks. Connected directly to an Aiven Cloud PostgreSQL instance, it utilizes custom PL/pgSQL mathematical operations for high-dimensional vector search.

The frontend is served as a premium, interactive glassmorphic chat console built with vanilla HTML5, CSS3, and JavaScript, utilizing vibrant styling, elegant micro-animations, and dynamic visual indicators.

---

## 🚀 Key Features

- **LLM-Driven Dynamic Router**: Classifies incoming queries deterministically into 5 specialized processing pathways:
  1. `DIRECT_LLM`: General conversations and standard IT explanations.
  2. `RAG_RETRIEVAL`: Semantic searching across internal IT policies, guidelines, and manuals using a local `SentenceTransformer` embedding model.
  3. `DB_QUERY`: Direct, secure, and read-only PostgreSQL queries (e.g., checking user statuses, roles, or counting logs).
  4. `TOOL_EXECUTION`: Running local modules like an arithmetic expression evaluator (`calculator`) or key-phrase crawler (`file_search`).
  5. `MULTI_SOURCE`: Multi-source coordination (e.g., executing a SQL query and retrieving RAG documents simultaneously).
- **Custom PL/pgSQL Vector Matcher**: Bypasses the need for `pgvector` extensions by registering custom vector `dot_product`, `magnitude`, and `cosine_similarity` math directly on vanilla PostgreSQL double-precision arrays.
- **FastAPI Modern Backend**: High-performance API endpoints, custom request logging middleware, and automatic schema integration.
- **Vibrant Glassmorphic Frontend**: Beautiful dashboard featuring:
  - Markdown message rendering
  - Expandable metadata accordion displaying the explicit routing decision, calculated latency, and underlying tools or SQL queries executed
  - Custom Toast notifications and real-time average latency tracking
  - RAG document ingestion panels to upload manuals directly into Aiven Cloud in real-time
  - Interactive "Quick-Start" preset query badges for testing

---

## 🛠️ System Architecture

```mermaid
graph TD
    User([User Chat Interface]) -->|HTTP POST /query| API[FastAPI Server]
    API -->|Route Query| Router{LLM Router}
    
    Router -->|DIRECT_LLM| Direct[Direct LLM Synthesis]
    Router -->|RAG_RETRIEVAL| RAG[Local Embeddings + Cosine Similarity SQL]
    Router -->|DB_QUERY| DB[Safe Read-Only SELECT Query]
    Router -->|TOOL_EXECUTION| Tools[Calculator / File Search Tools]
    Router -->|MULTI_SOURCE| Multi[Coordinated RAG + SQL DB Path]

    RAG --> Postgres[(PostgreSQL Database)]
    DB --> Postgres
    Multi --> Postgres
    
    Direct --> Synth[Final Answer Synthesizer]
    RAG --> Synth
    DB --> Synth
    Tools --> Synth
    Multi --> Synth

    Synth -->|Save Audit Log| Log[(PostgreSQL Queries Table)]
    Synth -->|JSON Response| API
    API -->|Display Bubble + Metadata| User
```

---

## 📂 Project Structure

```
├── agent/
│   ├── __init__.py
│   └── router.py             # Orchestrates LLM routing, SQL execution, and synthesis
├── api/
│   ├── __init__.py
│   ├── endpoints.py          # FastAPI endpoints (/query, /ingest, /eval)
│   └── main.py               # Main API launch script
├── core/
│   ├── __init__.py
│   ├── config.py             # Configuration schema (Pydantic Settings)
│   └── logging.py            # Custom request/response logger middleware
├── db/
│   ├── __init__.py
│   ├── init_db.py            # Database schema creator and initial mock seeder
│   ├── models.py             # SQLAlchemy models (User, Document, QueryLog)
│   └── session.py            # Engine initialization and session dependency
├── rag/
│   ├── __init__.py
│   └── pipeline.py           # Document chunking, local embedding, and custom SQL RAG
├── tools/
│   ├── __init__.py           # Tool registry and handler mapping
│   ├── base.py               # Base class definitions for extensible tools
│   ├── calculator.py         # Advanced mathematical formula executor
│   └── file_search.py        # Text and keyword crawler in workspace repositories
├── .gitignore                # Safeguards keys/databases from public git history
├── .env                      # API keys and connection strings (excluded from commit)
├── index.html                # Premium chat interface served on "/"
├── test_helpdesk.py          # Automated 10-query test and metric diagnostic suite
└── test_models.py            # Script verifying Gemini connection and model availability
```

---

## ⚙️ Prerequisites & Setup

### 1. Configure Environment Variables (`.env`)
Create a `.env` file in the root directory and specify the following variables:
```env
GEMINI_API_KEY=your-gemini-api-key
GROQ_API_KEY=your-groq-api-key
LLM_MODEL_NAME=llama-3.3-70b-versatile
DATABASE_URL=postgresql://user:password@host:port/dbname?sslmode=require
PORT=8000
ENVIRONMENT=development
```

### 2. Install Dependencies
```bash
pip install fastapi uvicorn sqlalchemy psycopg2-binary sentence-transformers groq google-genai pydantic-settings python-dotenv watchdog
```

---

## 🚀 Running the System

### 1. Initialize and Seed the Database
Register custom PL/pgSQL cosine similarity functions, create relational schemas, and seed mock users/documents into Aiven Cloud:
```bash
python -m db.init_db
```

### 2. Start the Backend Server
```bash
python -m api.main
```
The server will start on [http://127.0.0.1:8000](http://127.0.0.1:8000) and automatically serve the premium interactive dashboard!

### 3. Run Automated Diagnostic Suite
Test all 5 pathways end-to-end and calculate latencies using the built-in diagnostic suite:
```bash
python test_helpdesk.py
```
