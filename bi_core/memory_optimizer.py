"""
Memory Optimization Module for Business Intelligence Platform
Implements FAISS-based vector search, caching, and memory-efficient data processing
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from datetime import datetime, timedelta
from cachetools import TTLCache, LRUCache
import hashlib
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

from bi_core.settings import settings
from bi_core.telemetry import get_logger

logger = get_logger(__name__)

class MemoryOptimizer:
    """Memory-efficient data processing and caching system"""
    
    def __init__(self, max_cache_size: int = 1000, ttl_seconds: int = 3600):
        self.ttl_cache = TTLCache(maxsize=max_cache_size, ttl=ttl_seconds)
        self.lru_cache = LRUCache(maxsize=max_cache_size // 2)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Vector search components (lightweight implementation without FAISS for now)
        self.document_embeddings = {}
        self.document_store = {}
        
        logger.info("Memory optimizer initialized")
    
    def get_cache_key(self, query: str, analysis_type: str = "") -> str:
        """Generate cache key for queries"""
        content = f"{query}:{analysis_type}".encode('utf-8')
        return hashlib.md5(content).hexdigest()
    
    def cache_result(self, key: str, result: Any, use_ttl: bool = True):
        """Cache analysis result"""
        try:
            if use_ttl:
                self.ttl_cache[key] = result
            else:
                self.lru_cache[key] = result
            logger.debug(f"Cached result for key: {key[:10]}...")
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Retrieve cached result"""
        try:
            # Check TTL cache first
            if key in self.ttl_cache:
                logger.debug(f"Cache hit (TTL) for key: {key[:10]}...")
                return self.ttl_cache[key]
            
            # Check LRU cache
            if key in self.lru_cache:
                logger.debug(f"Cache hit (LRU) for key: {key[:10]}...")
                return self.lru_cache[key]
            
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve cached result: {e}")
            return None
    
    def optimize_data_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize data structures for memory efficiency"""
        try:
            # Remove redundant fields
            optimized = {}
            
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 1000:
                    # Truncate very long strings but keep essential info
                    optimized[key] = value[:500] + "..." + value[-100:] if len(value) > 600 else value
                elif isinstance(value, list) and len(value) > 20:
                    # Limit list size
                    optimized[key] = value[:20]
                elif isinstance(value, dict):
                    # Recursively optimize nested dictionaries
                    optimized[key] = self.optimize_data_structure(value)
                else:
                    optimized[key] = value
            
            return optimized
        except Exception as e:
            logger.error(f"Failed to optimize data structure: {e}")
            return data
    
    def batch_process_documents(self, documents: List[Dict[str, Any]], batch_size: int = 10) -> List[Dict[str, Any]]:
        """Process documents in memory-efficient batches"""
        processed_docs = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Process each batch
            for doc in batch:
                try:
                    # Optimize document structure
                    optimized_doc = self.optimize_data_structure(doc)
                    processed_docs.append(optimized_doc)
                except Exception as e:
                    logger.error(f"Failed to process document: {e}")
                    continue
        
        logger.info(f"Processed {len(processed_docs)} documents in batches")
        return processed_docs
    
    def similarity_search(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Lightweight similarity search without heavy vector dependencies"""
        try:
            # Simple keyword-based similarity for now
            query_words = set(query.lower().split())
            
            doc_scores = []
            for i, doc in enumerate(documents):
                content = str(doc.get('content', '') + ' ' + doc.get('title', '') + ' ' + doc.get('snippet', '')).lower()
                content_words = set(content.split())
                
                # Calculate Jaccard similarity
                intersection = len(query_words & content_words)
                union = len(query_words | content_words)
                similarity = intersection / union if union > 0 else 0
                
                # Boost score for title matches
                title = doc.get('title', '').lower()
                if any(word in title for word in query_words):
                    similarity += 0.2
                
                doc_scores.append((similarity, i, doc))
            
            # Sort by similarity and return top_k
            doc_scores.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, _, doc in doc_scores[:top_k]]
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return documents[:top_k]  # Fallback to first k documents
    
    def compress_data(self, data: Any) -> bytes:
        """Compress data for efficient storage"""
        try:
            import gzip
            json_str = json.dumps(data, separators=(',', ':'))
            return gzip.compress(json_str.encode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to compress data: {e}")
            return json.dumps(data).encode('utf-8')
    
    def decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress stored data"""
        try:
            import gzip
            json_str = gzip.decompress(compressed_data).decode('utf-8')
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to decompress data: {e}")
            return {}
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "ttl_cache_size": len(self.ttl_cache),
            "lru_cache_size": len(self.lru_cache),
            "document_store_size": len(self.document_store),
            "timestamp": datetime.now().isoformat()
        }
    
    def cleanup_memory(self):
        """Clean up memory by removing old cache entries"""
        try:
            # Clear old entries
            self.ttl_cache.clear()
            initial_lru_size = len(self.lru_cache)
            
            # Keep only recent LRU entries
            if len(self.lru_cache) > 100:
                # Create new cache with top items
                items = list(self.lru_cache.items())
                self.lru_cache.clear()
                for key, value in items[-50:]:  # Keep last 50 items
                    self.lru_cache[key] = value
            
            logger.info(f"Memory cleanup completed. LRU cache reduced from {initial_lru_size} to {len(self.lru_cache)}")
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")

class AsyncDataProcessor:
    """Asynchronous data processing for improved performance"""
    
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={'User-Agent': settings.user_agent}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_url_async(self, url: str) -> Optional[str]:
        """Asynchronously fetch URL content"""
        async with self.semaphore:
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        return content[:10000]  # Limit content size
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {e}")
                return None
    
    async def process_urls_batch(self, urls: List[str]) -> List[Tuple[str, Optional[str]]]:
        """Process multiple URLs concurrently"""
        tasks = [self.fetch_url_async(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                logger.error(f"Exception for {url}: {result}")
                processed_results.append((url, None))
            else:
                processed_results.append((url, result))
        
        return processed_results

# Global memory optimizer instance
memory_optimizer = MemoryOptimizer()

def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance"""
    return memory_optimizer

# Export main classes and functions
__all__ = [
    "MemoryOptimizer",
    "AsyncDataProcessor", 
    "memory_optimizer",
    "get_memory_optimizer"
]