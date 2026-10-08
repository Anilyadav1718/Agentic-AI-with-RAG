# ⚡ Nexora AI — Agentic AI Assistant with Hybrid RAG

**Nexora AI** is an Agentic AI assistant that combines **LLM tool calling, real-time web search, conversational context, and Hybrid Retrieval-Augmented Generation (RAG)**.

Instead of relying only on an LLM, Nexora AI dynamically decides whether to answer directly or invoke an external tool based on the user's intent.

---

## 🚀 Key Features

- 🤖 **Agentic Tool Calling** — dynamically selects tools based on user intent
- 🔁 **Multi-Step Agent Loop** — executes tools and feeds observations back to the LLM
- 🌐 **Real-Time Web Search** — retrieves current information using DDGS
- 📄 **PDF Knowledge Base** — upload and query PDF documents
- 🧠 **Semantic Search** — embedding-based retrieval using Sentence Transformers
- 🔤 **BM25 Search** — keyword-based retrieval for exact terms and technical keywords
- 🔀 **Hybrid RAG** — combines semantic and keyword retrieval
- 🏆 **Reciprocal Rank Fusion (RRF)** — merges multiple retrieval rankings
- 🗄️ **ChromaDB** — persistent vector storage
- 💬 **Conversation Context** — maintains chat and tool-use history
- 📚 **Source Transparency** — displays retrieved pages and ranking metrics

---

## 🏗️ Architecture

```text
                         User
                          │
                          ▼
                  ┌───────────────┐
                  │ Streamlit UI  │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  Nexora Agent │
                  │  Groq + Qwen  │
                  └───────┬───────┘
                          │
                    Tool Routing
              ┌───────────┼───────────┐
              ▼           ▼           ▼
         Calculator   Web Search   Knowledge Base
                                      │
                               ┌──────┴──────┐
                               ▼             ▼
                         Semantic Search    BM25
                               │             │
                               └──────┬──────┘
                                      ▼
                                     RRF
                                      │
                                      ▼
                                Top-K Context
                                      │
                                      ▼
                                 LLM Response
```

---

## 🧠 Agent Workflow

Nexora AI uses native LLM function calling to decide when external tools are required.

```text
User Query
    │
    ▼
LLM Intent Analysis
    │
    ▼
Tool Required?
 ┌──┴─────────┐
 │            │
 No          Yes
 │            │
 ▼            ▼
Answer     Select Tool
              │
              ▼
         Execute Tool
              │
              ▼
         Tool Result
              │
              ▼
        Return to LLM
              │
              ▼
         Final Answer
```

The current agent provides three tools:

| Tool | Purpose |
|---|---|
| `calculator` | Deterministic arithmetic operations |
| `web_search` | Retrieve current/public web information |
| `knowledge_base_search` | Search uploaded PDFs using Hybrid RAG |

---

## 📚 Hybrid RAG Pipeline

The document retrieval system combines **semantic search and BM25 keyword search**.

```text
PDF Upload
    │
    ▼
Text Extraction
    │
    ▼
Chunking
    │
    ▼
Embedding Generation
    │
    ▼
ChromaDB
    │
    ▼
User Query
   ┌┴──────────────────┐
   ▼                   ▼
Semantic Search     BM25 Search
   │                   │
   └─────────┬─────────┘
             ▼
    Reciprocal Rank Fusion
             │
             ▼
        Top 3 Chunks
             │
             ▼
            LLM
             │
             ▼
     Grounded Response
```

### Why Hybrid Search?

**Semantic Search** retrieves information based on meaning and contextual similarity.

**BM25** provides strong exact-keyword matching for technical terms, acronyms, and specific phrases.

**Reciprocal Rank Fusion (RRF)** combines both rankings to produce a stronger final retrieval result.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| LLM | Qwen |
| LLM Inference | Groq |
| UI | Streamlit |
| Agent Architecture | Native Function Calling |
| Embeddings | `all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| Semantic Retrieval | Cosine Similarity |
| Keyword Retrieval | BM25 |
| Rank Fusion | Reciprocal Rank Fusion |
| PDF Processing | PyPDF |
| Web Search | DDGS |
| Package Manager | uv |

---

## 📂 Project Structure

```text
local-ai-assistant/
│
├── app.py
├── llm.py
│
├── rag/
│   ├── __init__.py
│   └── document_rag.py
│
├── tools/
│   ├── __init__.py
│   ├── calculator.py
│   ├── web_search.py
│   └── registry.py
│
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Anilyadav1718/Agentic-AI-with-RAG.git

cd Agentic-AI-with-RAG
```

### 2. Install dependencies

Using `uv`:

```bash
uv sync
```

### 3. Configure Groq API Key

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

> Never commit `.env` or API keys to GitHub.

### 4. Run the application

```bash
uv run streamlit run app.py
```

Open the Streamlit URL displayed in the terminal.

---

## 💡 Example Queries

```text
What is Agentic AI?
```

→ Direct LLM response

```text
Calculate 9382 * 72
```

→ Calculator tool

```text
What is the latest AI news today?
```

→ Web Search tool

```text
According to my uploaded PDF, explain RAG.
```

→ Hybrid RAG knowledge-base tool

---

## 🔬 Technical Concepts Demonstrated

`Agentic AI` · `LLM Function Calling` · `Tool Routing` · `Multi-Step Agents` · `RAG` · `Embeddings` · `Vector Search` · `BM25` · `Reciprocal Rank Fusion` · `Hybrid Retrieval` · `Prompt Engineering` · `Context Grounding`

---

## ⚠️ Current Limitations

- PDF/vector storage is local to the application instance.
- Streamlit Community Cloud storage may reset after redeployment.
- Conversation memory is session-based rather than persistent.
- LLM usage is subject to Groq API rate limits.
- The current implementation is designed primarily as an engineering/demo project rather than a production multi-user system.

---

## 🔮 Future Improvements

- LangGraph-based agent orchestration
- Persistent conversational memory
- External production vector database
- Multiple-document RAG
- Query rewriting and reranking
- Streaming LLM responses
- Additional API/database tools
- Agent observability and tracing

---

## 👨‍💻 Author

**Anil Panchitha**

AI/ML | Agentic AI | RAG | Python | Automation

---

⭐ If you find this project useful, consider starring the repository.
