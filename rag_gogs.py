#!/usr/bin/env python3
import os
import re
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

# Third-party imports
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import MetadataQuery
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration constants
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_INDEX = os.getenv("WEAVIATE_INDEX", "Code")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
SEARCH_RESULT_LIMIT = int(os.getenv("SEARCH_RESULT_LIMIT", "10"))

class SimpleRAG:
    """Simplified RAG system for code repositories"""
    
    def __init__(self):
        self.client = self._connect_weaviate()
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.1)
        
        # Simple prompt template
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful code assistant. Based on the following documents, answer the user's question.

RELEVANT DOCUMENTS:
{context}

QUESTION: {question}

Instructions:
- Provide clear, accurate answers based on the retrieved documents
- Use proper markdown formatting for code
- Include file paths when referencing specific code
- If information is incomplete, say so clearly

Answer:"""
        )
    
    def _connect_weaviate(self) -> weaviate.WeaviateClient:
        """Connect to Weaviate"""
        try:
            parsed = urlparse(WEAVIATE_URL)
            host = parsed.hostname or "localhost"
            port = parsed.port or 8080
            
            client = weaviate.connect_to_local(
                host=host,
                port=port,
                grpc_port=50051,
                additional_config=AdditionalConfig(
                    timeout=Timeout(init=10, query=60, insert=120)
                )
            )
            
            if not client.is_ready():
                raise ConnectionError("Weaviate client is not ready")
                
            logger.info("Connected to Weaviate successfully")
            return client
            
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise
    
    def search_documents(self, query: str, limit: int = SEARCH_RESULT_LIMIT) -> List[Document]:
        """Search for relevant documents using BM25"""
        try:
            collection = self.client.collections.get(WEAVIATE_INDEX)
            
            # Try BM25 search first
            response = collection.query.bm25(
                query=query,
                limit=limit,
                return_metadata=MetadataQuery(score=True)
            )
            
            results = []
            for obj in response.objects:
                doc = Document(
                    page_content=obj.properties.get("text", ""),
                    metadata=obj.properties
                )
                results.append(doc)
            
            logger.info(f"Found {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def format_documents(self, documents: List[Document]) -> str:
        """Format documents for the LLM prompt"""
        if not documents:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            path = doc.metadata.get("path", "unknown")
            content = doc.page_content[:1000]  # Limit content length
            
            formatted_docs.append(f"[{i}] File: {path}\nContent:\n{content}\n")
        
        return "\n".join(formatted_docs)
    
    def ask(self, query: str) -> str:
        """Process a query and return an answer"""
        try:
            logger.info(f"Processing query: {query[:50]}...")
            
            # Search for relevant documents
            documents = self.search_documents(query)
            
            if not documents:
                return "I couldn't find any relevant documents for your query. Please try rephrasing your question."
            
            # Format documents for context
            context = self.format_documents(documents)
            
            # Generate response using LLM
            response = self.llm.invoke(
                self.prompt.format(context=context, question=query)
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"I encountered an error: {str(e)}"
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get basic information about the collection"""
        try:
            collection = self.client.collections.get(WEAVIATE_INDEX)
            
            # Get collection properties
            response = collection.query.fetch_objects(limit=1)
            
            info = {
                "collection_name": WEAVIATE_INDEX,
                "has_data": len(response.objects) > 0,
                "sample_properties": list(response.objects[0].properties.keys()) if response.objects else []
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}

def main():
    """Main CLI interface"""
    print("🚀 Simple Git RAG System")
    print("Type 'help' for commands, 'info' for collection info, 'exit' to quit.\n")
    
    try:
        rag = SimpleRAG()
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return
    
    while True:
        try:
            query = input("\nrag> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break
        
        if not query:
            continue
            
        if query.lower() in {"exit", "quit", "q"}:
            print("👋 Goodbye!")
            break
            
        elif query.lower() == "help":
            print("""
Available commands:
- help: Show this help message
- info: Show collection information
- exit/quit/q: Exit the system

Ask questions about your code repositories!
Examples:
- "What is joplin_weaviate_ollama?"
- "Show me Python files"
- "How do I run this project?"
            """)
            continue
            
        elif query.lower() == "info":
            info = rag.get_collection_info()
            print(f"\n📊 Collection Info:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            continue
        
        # Process the query
        print("\n🔍 Searching...")
        response = rag.ask(query)
        print(f"\n📝 Response:\n{response}")

if __name__ == "__main__":
    main()
