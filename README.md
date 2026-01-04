# RAGify

A production-ready Retrieval-Augmented Generation (RAG) system built with modern technologies to enable intelligent document processing, semantic search, and AI-powered question answering.

## 🏗️ Architecture Overview

RAGify is built on a robust, scalable architecture combining multiple cutting-edge technologies:

### Core Technologies Stack

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                     │
│                    (Async REST API Server)                   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   MongoDB    │    │   Qdrant     │    │  LLM APIs    │
│  (Metadata)  │    │  (Vectors)   │    │ (Generation) │
└──────────────┘    └──────────────┘    └──────────────┘
```

### 1. **FastAPI Framework**
The application is built on FastAPI, a modern, high-performance web framework for building APIs with Python 3.8+.

**Why FastAPI?**
- **Async Support**: Native async/await for handling concurrent requests efficiently
- **Automatic API Documentation**: Built-in Swagger UI and ReDoc
- **Type Safety**: Pydantic models for request/response validation
- **High Performance**: One of the fastest Python frameworks available

**Key Features:**
- RESTful API endpoints for document upload, processing, and querying
- Async file handling for large document uploads
- Automatic request validation and serialization
- Interactive API documentation at `/docs`

### 2. **MongoDB - Document Database**
MongoDB serves as the primary database for storing metadata and structured data.

**What MongoDB Stores:**
- **Projects Collection**: Project metadata and configuration
  - Project IDs, names, creation dates
  - Project-level settings and permissions
  
- **Assets Collection**: Uploaded file information
  - File names, sizes, types (PDF, TXT)
  - Upload timestamps and project associations
  - File storage paths
  
- **Chunks Collection**: Processed document chunks
  - Text content split into manageable pieces
  - Chunk metadata (order, source file, page numbers)
  - References to embeddings in vector database

**Why MongoDB?**
- Flexible schema for diverse document types
- Excellent performance for metadata queries
- Easy integration with Python via Motor (async driver)
- Scalable for growing document collections

**Connection Details:**
- Default URL: `mongodb://localhost:27017`
- Database Name: `mini-rag`
- Async operations using Motor (AsyncIOMotorClient)

### 3. **Qdrant - Vector Database**
Qdrant is a high-performance vector database used for semantic search and similarity matching.

**How It Works:**
1. **Document Embedding**: Text chunks are converted to 768-dimensional vectors using Gemini's embedding model
2. **Vector Storage**: Embeddings are stored in Qdrant collections (one per project)
3. **Semantic Search**: User queries are embedded and matched against stored vectors
4. **Similarity Scoring**: Cosine similarity finds the most relevant document chunks

**Key Features:**
- **Fast Vector Search**: Optimized for high-dimensional vector similarity
- **Collection-Based Organization**: Each project has its own vector collection
- **Batch Operations**: Efficient bulk insertion of embeddings
- **Distance Metrics**: Cosine similarity for semantic matching

**Configuration:**
- Storage Path: `qdrant_db` (local directory)
- Distance Method: Cosine similarity
- Embedding Size: 768 dimensions

### 4. **LLM Providers - AI Generation**
The system supports multiple Large Language Model providers with a flexible factory pattern.

#### **Supported Providers:**

**a) OpenAI**
- **Use Case**: Text generation and completion
- **Models**: GPT-3.5, GPT-4, or custom models via API
- **Configuration**: API key and optional custom endpoint (for local models like Ollama)

**b) Google Gemini** (Primary Embedding Provider)
- **Use Case**: 
  - Text embeddings (embedding-001 model)
  - Text generation (gemini-1.5-flash)
- **Embedding Size**: 768 dimensions
- **API**: Google AI Studio API

**c) Cohere**
- **Use Case**: Alternative for generation and embeddings
- **Features**: Specialized for search and RAG applications
- **API**: Cohere API

#### **LLM Architecture:**
```
LLMProviderFactory
    │
    ├── OpenAIProvider
    │   ├── generate_text()
    │   └── embed_text()
    │
    ├── GeminiProvider
    │   ├── generate_text()
    │   └── embed_text()
    │
    └── CoHereProvider
        ├── generate_text()
        └── embed_text()
```

**Flexible Configuration:**
- **Generation Backend**: Configurable (OpenAI, Gemini, Cohere)
- **Embedding Backend**: Configurable independently
- **Model Selection**: Specify exact model IDs
- **Temperature Control**: Adjust creativity (0.0 - 1.0)
- **Token Limits**: Control output length

## 🔄 RAG Pipeline Workflow

### Complete Document Processing Flow:

```
1. UPLOAD
   │
   ├─→ User uploads PDF/TXT file
   ├─→ File validation (type, size)
   ├─→ Store in project directory
   └─→ Create asset record in MongoDB
   
2. PROCESS
   │
   ├─→ Load file content (PyMuPDF for PDF, TextLoader for TXT)
   ├─→ Split into chunks (RecursiveCharacterTextSplitter)
   │   • Configurable chunk size (default: 100 chars)
   │   • Configurable overlap (default: 20 chars)
   ├─→ Store chunks in MongoDB with metadata
   └─→ Return chunk count
   
3. INDEX
   │
   ├─→ Retrieve chunks from MongoDB
   ├─→ Generate embeddings using Gemini
   │   • Batch processing for efficiency
   │   • 768-dimensional vectors
   ├─→ Create/reset Qdrant collection
   ├─→ Insert vectors with metadata
   └─→ Return indexing status
   
4. QUERY (RAG)
   │
   ├─→ User submits question
   ├─→ Embed question using same model
   ├─→ Search Qdrant for similar vectors
   │   • Configurable limit (default: 10)
   │   • Similarity threshold (default: 0.5)
   ├─→ Retrieve top matching chunks
   ├─→ Build context from retrieved chunks
   ├─→ Generate answer using LLM
   │   • Prompt template (AR/EN support)
   │   • Context + Question → Answer
   └─→ Return answer with sources
```

### API Endpoints:

**Data Management:**
- `POST /upload/{project_id}` - Upload documents
- `POST /process/{project_id}` - Process and chunk documents

**NLP Operations:**
- `POST /index/{project_id}` - Index documents into vector database
- `GET /index/info/{project_id}` - Get collection information
- `POST /search/{project_id}` - Semantic search
- `POST /rag/{project_id}` - Answer questions using RAG
- `POST /rag/all` - Query across all projects

## 🎯 Key Features

### 1. **Multi-Provider LLM Support**
Switch between OpenAI, Gemini, and Cohere without code changes. Configure via environment variables.

### 2. **Intelligent Document Chunking**
- Uses LangChain's RecursiveCharacterTextSplitter
- Maintains context with configurable overlap
- Preserves metadata (page numbers, source files)

### 3. **Project-Based Organization**
- Isolate documents by project
- Separate vector collections per project
- Query single or multiple projects

### 4. **Semantic Search**
- Vector similarity search using Qdrant
- Cosine similarity scoring
- Configurable result limits and thresholds

### 5. **Multi-Language Support**
- Template-based prompts in Arabic and English
- Configurable primary and default languages
- Easy to extend with new languages

### 6. **Batch Processing**
- Efficient bulk operations for embeddings
- Batch insertion into vector database
- Optimized for large document sets

### 7. **File Format Support**
- **PDF**: Full text extraction with PyMuPDF
- **TXT**: UTF-8 encoded text files
- Extensible for additional formats

## 📋 Prerequisites

Before running the application, ensure you have:

- **Python 3.8+**
- **MongoDB** (running on `localhost:27017`)
- **Qdrant** (local instance or cloud)
- **API Keys**:
  - Google Gemini API key (required for embeddings)
  - OpenAI API key (optional, for generation)
  - Cohere API key (optional, alternative provider)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd mini-rag
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Additional Requirements
The project uses LangChain for document processing. Ensure these are installed:
```bash
pip install langchain-community pymupdf motor pydantic-settings qdrant-client
```

### 4. Set Up MongoDB
Ensure MongoDB is running:
```bash
# Check if MongoDB is running
mongosh --eval "db.version()"

# If not installed, install MongoDB Community Edition
# Windows: Download from https://www.mongodb.com/try/download/community
# Linux: sudo apt-get install mongodb
# macOS: brew install mongodb-community
```

### 5. Set Up Qdrant
You can use Qdrant locally or in the cloud:

**Option A: Local Qdrant (Docker)**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Option B: Local Qdrant (Binary)**
Download from https://qdrant.tech/documentation/quick-start/

### 6. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Application Settings
APP_NAME="RAGify"
APP_VERSION="0.1"

# File Upload Settings
FILE_ALLOWED_TYPES=["text/plain","application/pdf"]
FILE_MAX_SIZE=10000  # in MB
FILE_DEFAULT_CHUNK_SIZE=512000  # 512KB

# MongoDB Configuration
MONGODB_URL="mongodb://localhost:27017"
MONGODB_DATABASE="mini-rag"

# LLM Configuration
GENERATION_BACKEND="OPENAI"  # Options: OPENAI, Gemini, COHERE
EMBEDDING_BACKEND="Gemini"   # Options: OPENAI, Gemini, COHERE

# API Keys (Get from respective providers)
OPENAI_API_KEY="your-openai-api-key"
OPENAI_API_URL="http://localhost:11434/v1/"  # For local models (Ollama)
COHERE_API_KEY="your-cohere-api-key"
Gemini_Api_key="your-gemini-api-key"  # Get from https://ai.google.dev/

# Model Configuration
GENERATION_MODEL_ID="gemini-1.5-flash"  # or "gpt-4", "aya-expanse:8b"
EMBEDDING_MODEL_ID="models/embedding-001"  # Gemini embedding model
EMBEDDING_MODEL_SIZE=768

# Generation Settings
INPUT_DAFAULT_MAX_CHARACTERS=1024
GENERATION_DAFAULT_MAX_TOKENS=200
GENERATION_DAFAULT_TEMPERATURE=0.1  # Lower = more focused, Higher = more creative

# Vector Database Configuration
VECTOR_DB_BACKEND="QDRANT"
VECTOR_DB_PATH="qdrant_db"  # Local storage path
VECTOR_DB_DISTANCE_METHOD="cosine"  # Options: cosine, dot

# Language Settings
PRIMARY_LANG="ar"  # Arabic
DEFAULT_LANG="ar"  # Fallback language
```

## 🚀 Running the Application

### Start the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

**Command Breakdown:**
- `main:app` - Run the `app` object from `main.py`
- `--reload` - Auto-reload on code changes (development mode)
- `--host 0.0.0.0` - Accept connections from any IP
- `--port 5000` - Run on port 5000

### Access the Application

Once running, you can access:

- **API Base URL**: `http://localhost:5000`
- **Swagger UI (Interactive Docs)**: `http://localhost:5000/docs`
- **ReDoc (Alternative Docs)**: `http://localhost:5000/redoc`

### Production Deployment

For production, use multiple workers:

```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --workers 4
```

Or use Gunicorn with Uvicorn workers:

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:5000
```

## 📖 Usage Examples

### 1. Upload a Document

```bash
curl -X POST "http://localhost:5000/upload/my-project" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "signal": "FILE_UPLOAD_SUCCESS",
  "file_id": "abc123_document.pdf"
}
```

### 2. Process Document into Chunks

```bash
curl -X POST "http://localhost:5000/process/my-project" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "abc123_document.pdf",
    "chunk_size": 500,
    "overlap_size": 50,
    "do_reset": 0
  }'
```

**Response:**
```json
{
  "signal": "PROCESSING_SUCCESS",
  "inserted_chunks": 45,
  "processed_files": 1
}
```

### 3. Index Documents (Create Embeddings)

```bash
curl -X POST "http://localhost:5000/index/my-project" \
  -H "Content-Type: application/json" \
  -d '{
    "do_reset": true
  }'
```

### 4. Ask a Question (RAG)

```bash
curl -X POST "http://localhost:5000/rag/my-project" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of the document?",
    "limit": 10,
    "threshold": 0.5
  }'
```

**Response:**
```json
{
  "answer": "The main topic discusses...",
  "sources": [
    {
      "chunk_text": "...",
      "score": 0.89,
      "metadata": {...}
    }
  ]
}
```

## 🏗️ Project Structure

```
mini-rag/
├── controllers/              # Business logic layer
│   ├── BaseController.py     # Base controller with common utilities
│   ├── DataController.py     # File upload and validation
│   ├── NLPController.py      # RAG and vector operations
│   ├── ProjectController.py  # Project management
│   └── ProcessController.py  # Document processing and chunking
│
├── models/                   # Data models and schemas
│   ├── db_schemes/           # MongoDB document schemas
│   │   ├── project.py        # Project schema
│   │   ├── asset.py          # Asset (file) schema
│   │   └── DataChunk.py      # Chunk schema
│   ├── enums/                # Enumerations
│   │   ├── DataBaseEnum.py   # Database collection names
│   │   ├── AssetTypeEnum.py  # Asset types
│   │   └── ProcessEnum.py    # File processing types
│   ├── ProjectModel.py       # Project CRUD operations
│   ├── AssetModel.py         # Asset CRUD operations
│   └── ChunkModel.py         # Chunk CRUD operations
│
├── routes/                   # API route definitions
│   ├── base.py               # Base routes (health check)
│   ├── data.py               # Upload and processing endpoints
│   ├── nlp.py                # RAG and search endpoints
│   └── schemes/              # Request/response schemas
│       ├── data.py           # Data endpoint schemas
│       └── nlp.py            # NLP endpoint schemas
│
├── stories/                  # Core functionality modules
│   ├── llm/                  # LLM provider implementations
│   │   ├── LLMInterface.py   # Abstract LLM interface
│   │   ├── LLMProviderFactory.py  # Factory pattern
│   │   ├── providers/
│   │   │   ├── OpenAIProvider.py
│   │   │   ├── GeminiProvider.py
│   │   │   └── CoHereProvider.py
│   │   └── templates/        # Prompt templates
│   │       ├── template_parser.py
│   │       └── locales/
│   │           ├── ar/       # Arabic templates
│   │           └── en/       # English templates
│   │
│   └── vectordb/             # Vector database implementations
│       ├── VectorDBInterface.py
│       ├── VectorDBProviderFactory.py
│       └── providers/
│           └── QdrantDBProvider.py
│
├── helpers/                  # Utility functions
│   └── config.py             # Configuration management (Pydantic)
│
├── uploads/                  # Uploaded files storage
│   └── [project-id]/         # Project-specific directories
│
├── qdrant_db/                # Qdrant local storage
│
├── main.py                   # Application entry point
├── .env                      # Environment configuration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔧 Configuration Details

### Supported File Types
- `text/plain` - Plain text files
- `application/pdf` - PDF documents

### Chunk Processing
- **Default Chunk Size**: 100 characters (configurable)
- **Default Overlap**: 20 characters (configurable)
- **Splitter**: RecursiveCharacterTextSplitter (LangChain)

### Vector Search
- **Default Limit**: 10 results
- **Default Threshold**: 0.5 similarity score
- **Distance Metric**: Cosine similarity

### LLM Generation
- **Default Max Tokens**: 200
- **Default Temperature**: 0.1 (focused responses)
- **Max Input Characters**: 1024

## 🌐 Multi-Language Support

The system includes template-based prompts for multiple languages:

**Supported Languages:**
- **Arabic (ar)**: Primary language
- **English (en)**: Default fallback

**Adding New Languages:**
1. Create directory: `stories/llm/templates/locales/{language_code}/`
2. Add `rag.py` with prompt templates
3. Update `PRIMARY_LANG` in `.env`

## 🔒 Security Considerations

- **API Keys**: Never commit `.env` file to version control
- **File Upload**: Validation for file types and sizes
- **Input Sanitization**: Clean file names and user inputs
- **MongoDB**: Use authentication in production
- **Rate Limiting**: Consider adding rate limits for API endpoints

## 🚀 Performance Optimization

- **Async Operations**: All I/O operations are async
- **Batch Processing**: Embeddings and vector insertions in batches
- **Connection Pooling**: MongoDB and Qdrant connection reuse
- **Caching**: Consider adding Redis for frequently accessed data

## 🐛 Troubleshooting

### MongoDB Connection Issues
```bash
# Check MongoDB status
sudo systemctl status mongod

# Restart MongoDB
sudo systemctl restart mongod
```

### Qdrant Connection Issues
```bash
# Check if Qdrant is running
curl http://localhost:6333/health

# Restart Qdrant (Docker)
docker restart qdrant
```

### API Key Errors
- Verify API keys in `.env` file
- Check key validity on provider websites
- Ensure no extra spaces or quotes

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📧 Contact

[Add contact information here]

## 🙏 Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/)
- [MongoDB](https://www.mongodb.com/)
- [Qdrant](https://qdrant.tech/)
- [LangChain](https://www.langchain.com/)
- [Google Gemini](https://ai.google.dev/)
- [OpenAI](https://openai.com/)
- [Cohere](https://cohere.ai/)
