# GOGS_WEAVIATE_OLLAMA RAG - Local LLM your Code 🚀

An enhanced Retrieval-Augmented Generation (RAG) system for Git repositories that provides intelligent code search and question answering capabilities. This system combines advanced document retrieval with conversational AI to help developers navigate and understand their codebases.

## Features ✨

### Core RAG Capabilities
- **Multi-Strategy Search**: Hybrid semantic + keyword search with intelligent routing
- **Context-Aware Responses**: Maintains conversation history and repository context
- **Intent Detection**: Automatically detects query intent (code search, documentation, debugging, etc.)
- **Smart Document Classification**: Categorizes files into source code, documentation, configuration, tests, etc.
- **Advanced Caching**: Response caching with TTL for improved performance
- **Real-time Statistics**: Track system usage and performance metrics

### Repository Synchronization
- **Multi-Instance Support**: Sync from multiple Gogs instances simultaneously
- **Incremental Updates**: Only processes changed files using intelligent caching
- **Flexible Filtering**: Include/exclude repositories and files using patterns and regex
- **Rate Limiting**: Respects API rate limits with configurable throttling
- **Concurrent Processing**: Parallel repository cloning and file processing
- **Error Recovery**: Robust error handling with retry mechanisms

## Architecture 🏗️

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gogs Repos    │    │   Local Mirror  │    │   Weaviate DB   │
│                 │───▶│                 │───▶│   (Vectors)     │
│ - Source Code   │    │ - Cloned Repos  │    │ - Embeddings    │
│ - Documentation │    │ - File Cache    │    │ - Metadata      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   RAG System    │◄───│   Smart Search  │
│                 │    │                 │    │                 │
│ - Natural Lang  │    │ - Intent Detect │    │ - Multi-Modal   │
│ - Code Questions│    │ - Context Mgmt  │    │ - Reranking     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Installation 📦

### Prerequisites
- Python 3.8+
- Docker (for Weaviate)
- Git
- Ollama (for local LLM)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd git-rag-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Weaviate
```bash
# Using Docker Compose (recommended)
docker-compose up -d weaviate

# Or run directly
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e DEFAULT_VECTORIZER_MODULE=none \
  -e ENABLE_MODULES=text2vec-transformers \
  -v weaviate_data:/var/lib/weaviate \
  semitechnologies/weaviate:latest
```

### 4. Install Ollama and Models
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3:8b
# Or use a smaller model for faster responses
ollama pull llama3:8b-instruct-q4_0
```

### 5. Configure Environment
Create a `.env` file:
```env
# Weaviate Configuration
WEAVIATE_URL=http://localhost:8080
WEAVIATE_INDEX=Documents

# Gogs Configuration  
GOGS_INSTANCES=https://gogs.example.com,http://internal-gogs:3000
GOGS_TOKENS={"gogs.example.com": "your-token-here", "internal-gogs:3000": "another-token"}

# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_MODEL=llama3:8b

# Processing Configuration
MAX_WORKERS=6
BATCH_SIZE=50
MAX_FILE_SIZE_MB=10
CACHE_TTL=3600

# File Filtering
INCLUDE_PATTERNS=*.py,*.js,*.md,*.txt,*.yml,*.json,*.go,*.java,*.cpp
EXCLUDE_PATTERNS=*.log,*.tmp,node_modules/*,.git/*,__pycache__/*

# Repository Filtering (optional)
REPO_INCLUDE_REGEX=^(company|project)-.*
REPO_EXCLUDE_REGEX=.*(test|temp|backup).*

# Performance
RATE_LIMIT_REQUESTS_PER_SECOND=10
MAX_CONTEXT_LENGTH=4000
SEARCH_RESULT_LIMIT=20
```

## Usage 🚀

### 1. Synchronize Repositories
First, sync your repositories from Gogs to the local vector database:

```bash
python sync_gogs.py
```

This will:
- Clone/update repositories from configured Gogs instances
- Process and extract content from source files
- Generate embeddings and store in Weaviate
- Cache results for incremental updates

### 2. Start the RAG System
```bash
python rag_gogs.py
```

### 3. Ask Questions
The system supports various types of queries:

#### Code Search
```
git-rag> Where is the user authentication function?
git-rag> Show me the database connection code
git-rag> Find the API endpoint for user registration
```

#### Documentation
```
git-rag> How do I build this project?
git-rag> What are the installation requirements?
git-rag> Explain the configuration options
```

#### Architecture & Design
```
git-rag> What is the overall architecture of this system?
git-rag> How are the modules organized?
git-rag> What design patterns are used?
```

#### Debugging & Troubleshooting
```
git-rag> Why might the login fail?
git-rag> What could cause database connection errors?
git-rag> How to debug API timeout issues?
```

### 4. System Commands
```bash
# Show system statistics
git-rag> stats

# Clear conversation context
git-rag> clear

# Show cache information
git-rag> cache

# Get help
git-rag> help

# Exit
git-rag> exit
```

## Configuration Options ⚙️

### Repository Synchronization
| Variable | Description | Default |
|----------|-------------|---------|
| `GOGS_INSTANCES` | Comma-separated list of Gogs URLs | `""` |
| `GOGS_TOKENS` | JSON mapping of host to access token | `{}` |
| `GOGS_MIRROR_DIR` | Local directory for repository mirrors | `gogs_mirrors` |
| `REPO_INCLUDE_REGEX` | Include repositories matching pattern | `None` |
| `REPO_EXCLUDE_REGEX` | Exclude repositories matching pattern | `None` |

### File Processing
| Variable | Description | Default |
|----------|-------------|---------|
| `INCLUDE_PATTERNS` | File patterns to include | `*.py,*.js,*.md,...` |
| `EXCLUDE_PATTERNS` | File patterns to exclude | `*.log,*.tmp,...` |
| `MAX_FILE_SIZE_MB` | Maximum file size to process | `10` |
| `MAX_WORKERS` | Concurrent processing threads | `6` |

### Search & Retrieval
| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL` | HuggingFace embedding model | `all-MiniLM-L6-v2` |
| `SEARCH_RESULT_LIMIT` | Maximum search results | `20` |
| `MAX_CONTEXT_LENGTH` | Maximum context tokens for LLM | `4000` |
| `CACHE_TTL` | Response cache TTL (seconds) | `3600` |

## Advanced Features 🔧

### Intent Detection
The system automatically detects query intent and optimizes search strategy:
- **Code Search**: Uses semantic + keyword search for function/class queries
- **Documentation**: Prioritizes semantic search for conceptual questions  
- **Build/Deploy**: Focuses on configuration and build files
- **Debug**: Emphasizes error handling and test files
- **Architecture**: Uses semantic search for design questions

### Context Management
- Maintains conversation history (last 20 exchanges)
- Tracks mentioned files and functions
- Remembers current repository context
- Provides contextual search result boosting

### Multi-Modal Search
- **Semantic Search**: Vector similarity using embeddings
- **Keyword Search**: BM25-based term matching
- **Hybrid Search**: Combines both approaches
- **Contextual Reranking**: Boosts results based on conversation context

### Performance Optimization
- **Incremental Sync**: Only processes changed files
- **Response Caching**: Caches answers with TTL
- **Batch Processing**: Efficient bulk operations
- **Rate Limiting**: Respectful API usage

## Monitoring & Debugging 📊

### View System Statistics
```bash
git-rag> stats
```
Shows:
- Conversation exchanges count
- Current repository context
- Mentioned files and functions
- Cache utilization
- Session duration

### Log Analysis
The system uses structured logging (JSON format) for easy analysis:
```bash
# View recent logs
tail -f logs/rag_system.log | jq '.'

# Filter by log level
grep '"level":"error"' logs/rag_system.log | jq '.'
```

### Performance Tuning
- Adjust `MAX_WORKERS` based on CPU cores
- Tune `BATCH_SIZE` for memory usage
- Modify `SEARCH_RESULT_LIMIT` for response quality vs speed
- Configure `CACHE_TTL` based on content update frequency

## Troubleshooting 🔧

### Common Issues

#### Weaviate Connection Error
```bash
# Check if Weaviate is running
curl http://localhost:8080/v1/meta

# Restart Weaviate
docker restart weaviate
```

#### Ollama Model Issues
```bash
# List available models
ollama list

# Pull required model
ollama pull llama3:8b

# Test model
ollama run llama3:8b "Hello world"
```

#### Memory Issues
- Reduce `MAX_WORKERS` and `BATCH_SIZE`
- Use smaller embedding model
- Limit `MAX_CONTEXT_LENGTH`
- Increase system swap space

#### Slow Performance
- Use GPU-enabled models if available
- Optimize `SEARCH_RESULT_LIMIT`
- Enable response caching
- Use SSD storage for repositories

### Debug Mode
Set environment variable for verbose logging:
```bash
export LOG_LEVEL=DEBUG
python rag_gogs.py
```

## API Integration 🔌

The system can be extended with a REST API:

```python
from flask import Flask, request, jsonify
from rag_gogs import EnhancedGitRAG

app = Flask(__name__)
rag = EnhancedGitRAG()

@app.route('/ask', methods=['POST'])
def ask_question():
    query = request.json.get('query')
    response = rag.ask(query)
    return jsonify({'answer': response})

@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify(rag.get_stats())
```

## Contributing 🤝

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black rag_gogs.py sync_gogs.py
flake8 rag_gogs.py sync_gogs.py
```

## License 📄

This project is licensed under the MIT License

## Acknowledgments 🙏

- [Weaviate](https://weaviate.io/) for the vector database
- [Ollama](https://ollama.ai/) for local LLM serving
- [LangChain](https://langchain.com/) for the RAG framework
- [HuggingFace](https://huggingface.co/) for embedding models
- [Sentence Transformers](https://www.sbert.net/) for semantic search

