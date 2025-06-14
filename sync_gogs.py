#!/usr/bin/env python3

from __future__ import annotations

import os
import re
import json
import argparse
import logging
import hashlib
import subprocess
import concurrent.futures as cf
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse
import urllib.request
import urllib.error

from dotenv import load_dotenv
import weaviate
from weaviate.config import AdditionalConfig, Timeout
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ───────────────────────── optional progress ──────────────────
try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # fallback noop if tqdm isn't installed
    tqdm = None  # pyright: ignore [reportGeneralTypeIssues]

# ───────────────────────── logging ─────────────────────────────
logging.basicConfig(
    filename="sync_gogs_errors.log",
    level=logging.ERROR,
    format="%(asctime)s  %(levelname)s  %(message)s",
)

# ───────────────────────── env vars ────────────────────────────
load_dotenv()
GOGS_INSTANCES = [url.strip() for url in os.getenv("GOGS_INSTANCES", "").split(",") if url.strip()]
GOGS_TOKENS = json.loads(os.getenv("GOGS_TOKENS", "{}"))
GOGS_MIRROR_DIR = Path(os.getenv("GOGS_MIRROR_DIR", "./gogs_mirrors"))
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
DEFAULT_INDEX = os.getenv("WEAVIATE_INDEX", "Documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CACHE_FILE = Path("gogs_cache.json")

# Supported text file extensions for indexing
TEXT_EXTENSIONS = {
    ".md", ".txt", ".py", ".js", ".html", ".css", ".json", ".yml", ".yaml",
    ".xml", ".sh", ".bash", ".sql", ".go", ".rs", ".java", ".cpp", ".c",
    ".h", ".hpp", ".cs", ".php", ".rb", ".pl", ".r", ".m", ".swift",
    ".kt", ".scala", ".clj", ".hs", ".ml", ".f90", ".pas", ".ada",
    ".tcl", ".lua", ".ps1", ".bat", ".cfg", ".conf", ".ini", ".env",
    ".dockerfile", ".makefile", ".cmake", ".gradle", ".maven", ".sbt"
}

# ────────────────── GOGS API helpers ──────────────────

def get_host_from_url(url: str) -> str:
    """Extract host:port from URL for token lookup."""
    parsed = urlparse(url)
    if parsed.port:
        return f"{parsed.hostname}:{parsed.port}"
    return parsed.hostname or url

def gogs_api_request(base_url: str, endpoint: str) -> Optional[List[Dict]]:
    """Make authenticated request to GOGS API."""
    host = get_host_from_url(base_url)
    token = GOGS_TOKENS.get(host)
    
    if not token:
        logging.error("No token found for host: %s", host)
        return None
    
    url = f"{base_url}/api/v1/{endpoint.lstrip('/')}"
    
    try:
        request = urllib.request.Request(url)
        request.add_header("Authorization", f"token {token}")
        request.add_header("Content-Type", "application/json")
        
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())
    
    except urllib.error.HTTPError as e:
        logging.error("HTTP error %d for %s: %s", e.code, url, e.reason)
        return None
    except Exception as e:
        logging.error("API request error for %s: %s", url, e)
        return None

def list_user_repos(base_url: str) -> List[Dict]:
    """Get all repositories for authenticated user."""
    repos = gogs_api_request(base_url, "/user/repos")
    return repos or []

# ────────────────── Git operations ──────────────────

def run_git_command(command: List[str], cwd: Path) -> bool:
    """Run git command and return success status."""
    try:
        result = subprocess.run(
            command, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        if result.returncode != 0:
            logging.error("Git command failed in %s: %s", cwd, result.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        logging.error("Git command timeout in %s: %s", cwd, " ".join(command))
        return False
    except Exception as e:
        logging.error("Git command error in %s: %s", cwd, e)
        return False

def clone_or_update_repo(repo_url: str, local_path: Path, token: str) -> bool:
    """Clone repository or update existing clone."""
    # Insert token into URL for authentication
    parsed = urlparse(repo_url)
    auth_url = f"{parsed.scheme}://{token}@{parsed.netloc}{parsed.path}"
    
    if local_path.exists():
        # Update existing repo
        print(f"🔄 Updating {local_path.name}")
        return (
            run_git_command(["git", "fetch", "origin"], local_path) and
            run_git_command(["git", "reset", "--hard", "origin/HEAD"], local_path)
        )
    else:
        # Clone new repo
        print(f"📥 Cloning {local_path.name}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        return run_git_command(["git", "clone", auth_url, str(local_path)], local_path.parent)

# ────────────────── Content extraction ──────────────────

def extract_text_content(file_path: Path) -> str:
    """Extract text content from supported file types."""
    try:
        # Try UTF-8 first, then fallback to latin-1
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return file_path.read_text(encoding="latin-1", errors="ignore")
    except Exception as e:
        logging.error("Text extraction error in %s: %s", file_path, e)
        return ""

def should_index_file(file_path: Path) -> bool:
    """Check if file should be indexed based on extension and size."""
    if file_path.suffix.lower() not in TEXT_EXTENSIONS:
        return False
    
    try:
        # Skip very large files (>1MB)
        if file_path.stat().st_size > 1024 * 1024:
            return False
        return True
    except Exception:
        return False

# ────────────────── Cache system ──────────────────

class RepoCache:
    """Cache for repository sync state and file hashes."""
    
    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = cache_file
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.load()
    
    def load(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                self._cache = json.loads(self.cache_file.read_text())
            except Exception as e:
                logging.error("Cache load error: %s", e)
                self._cache = {}
        else:
            self._cache = {}
    
    def save(self):
        """Save cache to disk."""
        try:
            self.cache_file.write_text(json.dumps(self._cache, indent=2))
        except Exception as e:
            logging.error("Cache save error: %s", e)
    
    def get_repo_hash(self, repo_path: Path) -> str:
        """Get current repository state hash."""
        try:
            # Use git rev-parse to get current commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logging.error("Git hash error in %s: %s", repo_path, e)
        return ""
    
    def repo_changed(self, repo_path: Path) -> bool:
        """Check if repository has changed since last sync."""
        repo_key = str(repo_path)
        current_hash = self.get_repo_hash(repo_path)
        
        if not current_hash:
            return True  # Assume changed if we can't get hash
        
        cached_hash = self._cache.get(repo_key, {}).get("commit_hash", "")
        return current_hash != cached_hash
    
    def update_repo(self, repo_path: Path):
        """Update cache entry for repository."""
        repo_key = str(repo_path)
        commit_hash = self.get_repo_hash(repo_path)
        
        if commit_hash:
            self._cache[repo_key] = {
                "commit_hash": commit_hash,
                "last_sync": datetime.now().isoformat(),
                "path": repo_key
            }
    
    def remove_missing(self, existing_repos: List[Path]):
        """Remove cache entries for repositories that no longer exist."""
        existing_str = {str(p) for p in existing_repos}
        to_remove = [repo for repo in self._cache.keys() if repo not in existing_str]
        for repo in to_remove:
            del self._cache[repo]

# ────────────────── Repository processing ──────────────────

RepoContent = Dict[str, Any]

def process_repository(repo_path: Path, cache: RepoCache) -> Optional[List[RepoContent]]:
    """Process repository and extract indexable content."""
    
    # Check if repo has changed
    if not cache.repo_changed(repo_path):
        return None
    
    if not repo_path.exists():
        return None
    
    contents = []
    repo_name = repo_path.name
    
    try:
        # Walk through repository files
        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Skip hidden files and directories
            if any(part.startswith('.') for part in file_path.parts):
                continue
            
            if not should_index_file(file_path):
                continue
            
            # Extract content
            text_content = extract_text_content(file_path)
            if not text_content.strip():
                continue
            
            # Get relative path within repo
            rel_path = file_path.relative_to(repo_path)
            
            # Create content entry
            content = {
                "title": f"{repo_name}/{rel_path}",
                "path": str(file_path),
                "relative_path": str(rel_path),
                "repository": repo_name,
                "content": text_content,
                "file_type": file_path.suffix.lower(),
                "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                "file_size": file_path.stat().st_size,
            }
            
            contents.append(content)
    
    except Exception as e:
        logging.error("Repository processing error in %s: %s", repo_path, e)
        return None
    
    # Update cache
    cache.update_repo(repo_path)
    
    return contents if contents else None

# ────────────────── Repository syncing ──────────────────

def sync_repositories(*, workers: int, show_progress: bool) -> List[RepoContent]:
    """Sync all configured GOGS repositories."""
    
    if not GOGS_INSTANCES:
        print("⚠️  No GOGS instances configured")
        return []
    
    # Initialize cache
    cache = RepoCache()
    
    # Ensure mirror directory exists
    GOGS_MIRROR_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect all repositories to sync
    all_repos = []
    repo_paths = []
    
    for base_url in GOGS_INSTANCES:
        print(f"🔍 Discovering repositories on {base_url}")
        host = get_host_from_url(base_url)
        token = GOGS_TOKENS.get(host)
        
        if not token:
            print(f"⚠️  No token for {host}, skipping")
            continue
        
        repos = list_user_repos(base_url)
        print(f"📊 Found {len(repos)} repositories on {host}")
        
        for repo in repos:
            repo_info = {
                "name": repo["name"],
                "clone_url": repo["clone_url"],
                "host": host,
                "token": token,
                "local_path": GOGS_MIRROR_DIR / host / repo["name"]
            }
            all_repos.append(repo_info)
            repo_paths.append(repo_info["local_path"])
    
    if not all_repos:
        print("📭 No repositories found")
        return []
    
    print(f"🔄 Syncing {len(all_repos)} repositories")
    
    # Clean up cache for missing repos
    cache.remove_missing(repo_paths)
    
    # Sync repositories
    sync_bar = tqdm(total=len(all_repos), desc="Syncing repos", unit="repo") if (show_progress and tqdm) else None
    
    for repo_info in all_repos:
        success = clone_or_update_repo(
            repo_info["clone_url"],
            repo_info["local_path"],
            repo_info["token"]
        )
        
        if not success:
            print(f"⚠️  Failed to sync {repo_info['name']}")
        
        if sync_bar:
            sync_bar.update(1)
    
    if sync_bar:
        sync_bar.close()
    
    # Process repositories for content
    print("📝 Processing repository content")
    
    # Filter repos that need processing
    changed_repos = [path for path in repo_paths if cache.repo_changed(path)]
    
    print(f"📊 Found {len(repo_paths)} total repos, {len(changed_repos)} changed")
    
    if not changed_repos:
        cache.save()
        return []
    
    process_bar = tqdm(total=len(changed_repos), desc="Processing", unit="repo") if (show_progress and tqdm) else None
    all_contents = []
    
    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_repository, repo_path, cache) for repo_path in changed_repos]
        
        for future in cf.as_completed(futures):
            result = future.result()
            if result:
                all_contents.extend(result)
            if process_bar:
                process_bar.update(1)
    
    if process_bar:
        process_bar.close()
    
    # Save cache after processing
    cache.save()
    
    return all_contents

# ───────────────────── Weaviate operations (same as Joplin) ──────────────────

def weaviate_client() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(
        host="localhost", port=8080, grpc_port=50051,
        additional_config=AdditionalConfig(timeout=Timeout(init=5)),
    )

def ensure_schema(client: weaviate.WeaviateClient, index_name: str):
    """Create the collection if it doesn't exist with BM25 + vector search support."""
    try:
        if client.collections.exists(index_name):
            print(f"✅ Collection '{index_name}' already exists")
            return
        
        # Try to configure BM25 if available
        inverted_config = None
        try:
            # Try newer API first
            if hasattr(Configure, 'inverted_index') and hasattr(Configure, 'BM25'):
                inverted_config = Configure.inverted_index(
                    bm25=Configure.BM25(b=0.75, k1=1.2)
                )
        except (AttributeError, TypeError):
            # Fall back to basic inverted index
            try:
                if hasattr(Configure, 'inverted_index'):
                    inverted_config = Configure.inverted_index()
            except (AttributeError, TypeError):
                inverted_config = None
        
        collection_config = {
            "name": index_name,
            "vectorizer_config": Configure.Vectorizer.none(),
            "properties": [
                Property(name="text", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="path", data_type=DataType.TEXT),
                Property(name="relative_path", data_type=DataType.TEXT),
                Property(name="repository", data_type=DataType.TEXT),
                Property(name="file_type", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="last_modified", data_type=DataType.DATE),
                Property(name="file_size", data_type=DataType.INT),
                Property(name="content_hash", data_type=DataType.TEXT),
            ],
        }
        
        if inverted_config:
            collection_config["inverted_index_config"] = inverted_config
        
        client.collections.create(**collection_config)
        
        config_desc = "with hybrid search support" if inverted_config else "with basic search support"
        print(f"✅ Created collection '{index_name}' {config_desc}")
        
    except Exception as e:
        logging.error("Schema creation error: %s", e)
        print(f"⚠️  Schema error: {e}")
        raise

def remove_existing_documents(client: weaviate.WeaviateClient, index_name: str, repositories: List[str]):
    """Remove existing documents for repositories being re-processed."""
    if not repositories:
        return
    
    try:
        collection = client.collections.get(index_name)
        
        for repo in repositories:
            collection.data.delete_many(
                where=Filter.by_property("repository").equal(repo)
            )
        
        print(f"🗑️  Removed existing documents for {len(repositories)} repositories")
        
    except Exception as e:
        logging.error("Document removal error: %s", e)
        print(f"⚠️  Document removal error: {e}")

def upload_batch(contents_batch: List[RepoContent], splitter, vectorstore) -> int:
    """Upload a batch of repository contents."""
    texts, metas = [], []
    
    for content in contents_batch:
        content_hash = hashlib.sha256(content["content"].encode()).hexdigest()
        
        doc_meta = {
            "title": content["title"],
            "path": content["path"],
            "relative_path": content["relative_path"],
            "repository": content["repository"],
            "file_type": content["file_type"],
            "source": "gogs",
            "last_modified": content.get("last_modified", ""),
            "file_size": content.get("file_size", 0),
            "content_hash": content_hash,
        }
        
        for chunk in splitter.split_text(content["content"]):
            texts.append(chunk)
            metas.append(doc_meta.copy())
    
    if texts:
        vectorstore.add_texts(texts, metas)
    
    return len(texts)

def upload(contents: List[RepoContent], *, index_name: str, show_progress: bool = False, batch_size: int = 1000):
    """Upload repository contents to Weaviate."""
    if not contents:
        print("⚠️  Nothing new to upload.")
        return
    
    total_contents = len(contents)
    total_batches = (total_contents + batch_size - 1) // batch_size
    print(f"📤 Uploading {total_contents} files → '{index_name}' in {total_batches} batch(es)…")
    
    client = None
    
    try:
        client = weaviate_client()
        ensure_schema(client, index_name)
        
        # Remove existing documents for updated repositories
        repositories = list(set(content["repository"] for content in contents))
        remove_existing_documents(client, index_name, repositories)
        
        embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        vectorstore = WeaviateVectorStore(
            client=client,
            index_name=index_name,
            text_key="text",
            embedding=embedder,
        )
        
        batch_bar = tqdm(total=total_batches, desc="Uploading", unit="batch") if (show_progress and tqdm) else None
        total_chunks = 0
        
        for i in range(0, total_contents, batch_size):
            batch = contents[i:i + batch_size]
            if batch_bar:
                batch_bar.set_description(f"Batch {(i // batch_size)+1}/{total_batches} ({len(batch)} files)")
            
            try:
                total_chunks += upload_batch(batch, splitter, vectorstore)
                if batch_bar:
                    batch_bar.update(1)
            except Exception as e:
                logging.error("Batch %d failed: %s", (i // batch_size)+1, e)
                print(f"⚠️  Batch {(i // batch_size)+1} failed: {e}")
                continue
        
        if batch_bar:
            batch_bar.close()
        
        print(f"✅ Upload complete. {total_chunks} chunks across {len(contents)} files.")
        
    finally:
        if client:
            client.close()

# ─────────────────────── Search functionality ───────────────────────────

def search_repositories(query: str, index_name: str = DEFAULT_INDEX, limit: int = 10, 
                       search_type: str = "hybrid", alpha: float = 0.7):
    """Search repository contents with multiple search modes."""
    client = None
    try:
        client = weaviate_client()
        collection = client.collections.get(index_name)
        
        if search_type == "hybrid":
            response = collection.query.hybrid(
                query=query,
                alpha=alpha,
                limit=limit,
                return_metadata=["score"]
            )
        elif search_type == "vector":
            embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            query_vector = embedder.embed_query(query)
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                return_metadata=["distance"]
            )
        elif search_type == "bm25":
            response = collection.query.bm25(
                query=query,
                limit=limit,
                return_metadata=["score"]
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        results = []
        for obj in response.objects:
            result = {
                "content": obj.properties.get("text", ""),
                "title": obj.properties.get("title", ""),
                "path": obj.properties.get("path", ""),
                "relative_path": obj.properties.get("relative_path", ""),
                "repository": obj.properties.get("repository", ""),
                "file_type": obj.properties.get("file_type", ""),
                "last_modified": obj.properties.get("last_modified", ""),
            }
            
            if hasattr(obj.metadata, 'score'):
                result["score"] = obj.metadata.score
            elif hasattr(obj.metadata, 'distance'):
                result["distance"] = obj.metadata.distance
            
            results.append(result)
        
        return results
        
    finally:
        if client:
            client.close()

# ─────────────────────── Cache management ───────────────────────────

def cache_info(cache_file: Path = CACHE_FILE):
    """Display cache statistics."""
    if not cache_file.exists():
        print("📊 Cache file does not exist yet")
        return
    
    try:
        cache_data = json.loads(cache_file.read_text())
        total_repos = len(cache_data)
        
        print(f"📊 GOGS Cache Information:")
        print(f"   • Total cached repositories: {total_repos}")
        print(f"   • Cache file size: {cache_file.stat().st_size / 1024:.1f} KB")
        
        # Check for orphaned entries
        orphaned = []
        for repo_path in cache_data.keys():
            if not Path(repo_path).exists():
                orphaned.append(repo_path)
        
        if orphaned:
            print(f"   • Orphaned entries: {len(orphaned)}")
        else:
            print(f"   • No orphaned entries found")
            
    except Exception as e:
        print(f"⚠️  Error reading cache: {e}")

def clean_cache(cache_file: Path = CACHE_FILE, dry_run: bool = True):
    """Clean orphaned entries from cache."""
    if not cache_file.exists():
        print("📊 Cache file does not exist")
        return
    
    try:
        cache_data = json.loads(cache_file.read_text())
        original_count = len(cache_data)
        
        to_remove = []
        for repo_path in cache_data.keys():
            if not Path(repo_path).exists():
                to_remove.append(repo_path)
        
        if not to_remove:
            print("✅ No orphaned entries to clean")
            return
        
        print(f"🧹 Found {len(to_remove)} orphaned entries")
        
        if dry_run:
            print("   (DRY RUN - use --clean-cache --force to actually remove)")
            for path in to_remove[:5]:
                print(f"   - {path}")
            if len(to_remove) > 5:
                print(f"   - ... and {len(to_remove) - 5} more")
        else:
            for path in to_remove:
                del cache_data[path]
            
            cache_file.write_text(json.dumps(cache_data, indent=2))
            print(f"✅ Removed {len(to_remove)} orphaned entries")
            print(f"   Cache size: {original_count} → {len(cache_data)} repositories")
            
    except Exception as e:
        print(f"⚠️  Error cleaning cache: {e}")

# ─────────────────────────── CLI ───────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Sync GOGS repositories to Weaviate")
    
    ap.add_argument("--sync", action="store_true", help="sync repositories from GOGS")
    ap.add_argument("--upload", action="store_true", help="upload to Weaviate")
    ap.add_argument("--search", type=str, help="search query")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="threads for processing (default: logical CPUs)")
    ap.add_argument("--progress", action="store_true", help="show progress bars")
    ap.add_argument("--batch-size", type=int, default=1000, help="files per upload batch")
    ap.add_argument("--index", default=DEFAULT_INDEX, help="Weaviate collection name")
    ap.add_argument("--search-type", choices=["hybrid", "vector", "bm25"], default="hybrid",
                    help="search mode (default: hybrid)")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="hybrid search balance: 0.0=pure BM25, 1.0=pure vector")
    ap.add_argument("--limit", type=int, default=10, help="max search results")
    
    # Cache management
    ap.add_argument("--cache-info", action="store_true", help="show cache statistics")
    ap.add_argument("--clean-cache", action="store_true", help="clean orphaned cache entries")
    ap.add_argument("--force", action="store_true", help="actually perform cache cleaning")
    
    args = ap.parse_args()
    
    # Cache management commands
    if args.cache_info:
        cache_info()
        return
    
    if args.clean_cache:
        clean_cache(dry_run=not args.force)
        return
    
    if args.search:
        print(f"🔍 Searching repositories for: '{args.search}' (mode: {args.search_type})")
        results = search_repositories(
            args.search,
            index_name=args.index,
            limit=args.limit,
            search_type=args.search_type,
            alpha=args.alpha
        )
        
        if not results:
            print("📭 No results found.")
        else:
            print(f"📊 Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                score_info = ""
                if "score" in result:
                    score_info = f" (score: {result['score']:.3f})"
                elif "distance" in result:
                    score_info = f" (distance: {result['distance']:.3f})"
                
                print(f"\n{i}. {result['title']}{score_info}")
                print(f"   📁 {result['repository']} | 🏷️ {result['file_type']}")
                print(f"   📄 {result['relative_path']}")
                
                # Show snippet of content
                content = result['content'][:200]
                if len(result['content']) > 200:
                    content += "..."
                print(f"   💬 {content}")
    
    elif args.sync:
        contents = sync_repositories(workers=args.workers, show_progress=args.progress)
        print(f"🔄 {len(contents)} new/changed files detected.")
        
        if args.upload and contents:
            upload(contents, index_name=args.index, show_progress=args.progress,
                   batch_size=args.batch_size)
    
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
