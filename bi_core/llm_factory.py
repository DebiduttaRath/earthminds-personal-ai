"""
LLM Factory for Business Intelligence Platform
Handles multiple LLM backends with intelligent routing
"""

from typing import Any, Dict, Optional
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
import requests
from bi_core.settings import settings
from bi_core.telemetry import get_logger

logger = get_logger(__name__)

class LLMFactory:
    """Factory class for creating and managing LLM instances"""
    
    def __init__(self):
        self._groq_client = None
        self._ollama_client = None
    
    def get_llm(self, for_reasoning: bool = False, backend: Optional[str] = None, **kwargs):
        """
        Returns a LangChain-compatible chat model based on configuration
        
        Args:
            for_reasoning: Whether this LLM will be used for complex reasoning tasks
            backend: Override the default backend selection
            **kwargs: Additional parameters for the model
        """
        target_backend = backend or settings.llm_backend
        
        try:
            if target_backend == "groq":
                return self._get_groq_llm(for_reasoning, **kwargs)
            elif target_backend == "deepseek":
                return self._get_deepseek_llm(for_reasoning, **kwargs)
            elif target_backend == "ollama":
                return self._get_ollama_llm(for_reasoning, **kwargs)
            else:
                raise ValueError(f"Unknown LLM backend: {target_backend}")
                
        except Exception as e:
            logger.error(f"Failed to initialize {target_backend} LLM: {e}")
            # Fallback to Ollama if available
            if target_backend != "ollama":
                logger.info("Attempting fallback to Ollama...")
                return self._get_ollama_llm(for_reasoning, **kwargs)
            raise
    
    def _get_groq_llm(self, for_reasoning: bool = False, **kwargs):
        """Initialize Groq LLM with business intelligence optimizations"""
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY not configured")
        
        # For business intelligence, we want to use reasoning-capable models
        model_name = settings.groq_model
        if for_reasoning and "deepseek-r1" not in model_name:
            # Prefer DeepSeek R1 for reasoning tasks
            model_name = "deepseek-r1-distill-llama-70b"
        
        config = {
            "temperature": kwargs.get("temperature", 0.6 if for_reasoning else 0.3),
            "max_tokens": kwargs.get("max_tokens", settings.max_output_tokens),
            "top_p": kwargs.get("top_p", 0.95),
        }
        
        # Enable reasoning format for DeepSeek R1
        if "deepseek-r1" in model_name:
            config["extra_body"] = {"reasoning_format": "raw"}
        
        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=model_name,
            **config
        )
        
        logger.info(f"Initialized Groq LLM: {model_name}")
        return llm
    
    def _get_deepseek_llm(self, for_reasoning: bool = False, **kwargs):
        """Initialize DeepSeek LLM via API"""
        if not settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY not configured")
        
        # Use R1 for reasoning, V3 for general chat
        model_name = "deepseek-r1" if for_reasoning else "deepseek-v3"
        
        # Note: This is a simplified implementation
        # In production, you'd use the official DeepSeek SDK
        config = {
            "temperature": kwargs.get("temperature", 0.1 if for_reasoning else 0.6),
            "max_tokens": kwargs.get("max_tokens", settings.max_output_tokens),
        }
        
        # For now, we'll use the OpenAI-compatible interface
        try:
            llm = init_chat_model(
                f"openai:{model_name}",
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                **config
            )
            logger.info(f"Initialized DeepSeek LLM: {model_name}")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek LLM: {e}")
            raise
    
    def _get_ollama_llm(self, for_reasoning: bool = False, **kwargs):
        """Initialize Ollama LLM for local inference"""
        model_name = settings.ollama_reason_model if for_reasoning else settings.ollama_chat_model
        
        config = {
            "temperature": kwargs.get("temperature", 0.1 if for_reasoning else 0.6),
            "num_predict": kwargs.get("max_tokens", settings.max_output_tokens),
        }
        
        llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=model_name,
            **config
        )
        
        logger.info(f"Initialized Ollama LLM: {model_name}")
        return llm
    
    def get_smart_llm(self, query: str, **kwargs):
        """
        Intelligently select LLM based on query complexity
        
        Args:
            query: The user query to analyze
            **kwargs: Additional parameters
        """
        reasoning_keywords = [
            "analyze", "compare", "evaluate", "assess", "reasoning", "chain of thought",
            "step by step", "explain why", "pros and cons", "advantages", "disadvantages",
            "investment", "financial analysis", "competitive analysis", "market research",
            "trend analysis", "strategic", "forecast", "predict", "recommendation"
        ]
        
        # Check if query requires complex reasoning
        requires_reasoning = any(keyword in query.lower() for keyword in reasoning_keywords)
        
        # Complex queries or long queries likely need reasoning
        if requires_reasoning or len(query.split()) > 20:
            logger.info("Using reasoning-capable LLM for complex query")
            return self.get_llm(for_reasoning=True, **kwargs)
        else:
            logger.info("Using fast inference LLM for simple query")
            return self.get_llm(for_reasoning=False, **kwargs)
    
    def health_check(self, backend: str = None) -> Dict[str, Any]:
        """Check the health of LLM backends"""
        target_backend = backend or settings.llm_backend
        
        try:
            if target_backend == "groq":
                # Test Groq connection
                llm = self._get_groq_llm()
                response = llm.invoke("Test connection")
                return {"status": "healthy", "backend": "groq", "response_length": len(str(response))}
                
            elif target_backend == "ollama":
                # Test Ollama connection
                response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
                return {"status": "healthy" if response.status_code == 200 else "unhealthy", "backend": "ollama"}
                
            elif target_backend == "deepseek":
                # Test DeepSeek connection
                llm = self._get_deepseek_llm()
                response = llm.invoke("Test connection")
                return {"status": "healthy", "backend": "deepseek", "response_length": len(str(response))}
                
        except Exception as e:
            return {"status": "unhealthy", "backend": target_backend, "error": str(e)}

# Global LLM factory instance
llm_factory = LLMFactory()

def get_llm(*args, **kwargs):
    """Convenience function to get LLM from global factory"""
    return llm_factory.get_llm(*args, **kwargs)

def get_smart_llm(*args, **kwargs):
    """Convenience function to get smart LLM selection"""
    return llm_factory.get_smart_llm(*args, **kwargs)
