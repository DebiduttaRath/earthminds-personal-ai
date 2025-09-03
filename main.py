"""
🚀 Business Intelligence Platform - Optimized & Production Ready
Unified AI-powered business analysis with enhanced web scraping and multi-LLM support
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import time
import logging
import traceback
import re
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage, SystemMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('business_intelligence.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import core components with fallback
try:
    from bi_core.graph import build_business_graph
    from bi_core.settings import settings
    from bi_core.business_workflows import BusinessIntelligenceWorkflow
    from bi_core.telemetry import setup_telemetry, get_logger
    from bi_core.memory_optimizer import memory_optimizer
    from bi_core.anti_hallucination import verify_analysis_reliability
    from bi_core.llm_factory import llm_factory
    CORE_AVAILABLE = True
    logger.info("✅ Core BI components loaded successfully")
except ImportError as e:
    CORE_AVAILABLE = False
    logger.warning(f"⚠️ Core BI components not available: {e}")

# Free model configurations for DeepSeek and Ollama
FREE_MODEL_CONFIG = {
    "deepseek": {
        "api_url": "https://api.deepseek.com/v1/chat/completions",
        "models": {
            "chat": "deepseek-v3",
            "reasoning": "deepseek-r1"
        },
        "free_tier": True
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "models": {
            "chat": "llama3.2:3b",  # Free local model
            "reasoning": "qwen2.5:7b"  # Free local reasoning model
        },
        "free_tier": True
    },
    "groq": {
        "models": {
            "fast": "llama-3.3-70b-versatile",
            "reasoning": "mixtral-8x7b-32768"
        },
        "free_tier": True,
        "rpm_limit": 30  # Free tier limit
    }
}

# Enhanced web scraping class for global deployment
class GlobalWebScrapingEngine:
    """Production-ready web scraping engine optimized for global deployment"""
    
    def __init__(self):
        self.session = requests.Session()
        # Optimize headers for better global compatibility
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none"
        })
        
        # Global proxy rotation (for AWS EC2 deployment)
        self.proxies = []
        self.current_proxy = 0
        
    def search_web_enhanced(self, query: str, max_results: int = 10, region: str = "us-en") -> List[Dict]:
        """Enhanced global web search with multiple fallbacks"""
        try:
            logger.info(f"Global web search for: {query} (region: {region})")
            results = []
            
            # Primary search with DuckDuckGo
            try:
                with DDGS() as ddgs:
                    search_results = list(ddgs.text(
                        query,
                        max_results=max_results,
                        region=region,
                        timelimit='m',  # Last month for freshest data
                        safesearch='moderate'
                    ))
                    
                    for result in search_results:
                        # Enhanced relevance scoring
                        business_score = self._calculate_business_relevance(result)
                        
                        results.append({
                            "title": result.get("title", ""),
                            "snippet": result.get("body", ""),
                            "url": result.get("href", ""),
                            "relevance_score": business_score,
                            "source": "duckduckgo",
                            "timestamp": datetime.now().isoformat()
                        })
                        
            except Exception as e:
                logger.warning(f"DuckDuckGo search failed: {e}")
                
                # Fallback search methods for better global coverage
                results.extend(self._fallback_search(query, max_results))
            
            # Sort by relevance and return
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            logger.info(f"Found {len(results)} global search results")
            return results
            
        except Exception as e:
            logger.error(f"Global web search failed: {e}")
            return [{"title": "Search Error", "snippet": f"Error: {str(e)}", "url": "", "relevance_score": 0}]
    
    def _calculate_business_relevance(self, result: Dict) -> float:
        """Calculate business relevance score with enhanced metrics"""
        score = 0.0
        
        title = result.get("title", "").lower()
        snippet = result.get("body", "").lower()
        url = result.get("href", "").lower()
        combined_text = f"{title} {snippet}"
        
        # Business keywords with weights
        business_keywords = {
            "financial": 3.0, "revenue": 2.5, "profit": 2.5, "earnings": 2.5,
            "market": 2.0, "company": 2.0, "business": 2.0, "industry": 2.0,
            "analysis": 2.0, "growth": 2.0, "investment": 2.0, "stock": 2.0,
            "competitive": 1.5, "strategy": 1.5, "performance": 1.5,
            "trends": 1.0, "data": 1.0, "report": 1.0
        }
        
        for keyword, weight in business_keywords.items():
            if keyword in combined_text:
                score += weight
        
        # Boost trusted sources
        trusted_domains = [
            "bloomberg.com", "reuters.com", "wsj.com", "ft.com", "marketwatch.com",
            "yahoo.com/finance", "sec.gov", "investopedia.com", "forbes.com",
            "cnbc.com", "businesswire.com", "prnewswire.com", "nasdaq.com"
        ]
        
        if any(domain in url for domain in trusted_domains):
            score *= 1.5
        
        # Penalize low-quality sources
        spam_indicators = ["ads", "spam", "affiliate", "clickbait"]
        if any(indicator in combined_text for indicator in spam_indicators):
            score *= 0.5
        
        return min(score, 10.0)  # Cap at 10.0
    
    def _fallback_search(self, query: str, max_results: int) -> List[Dict]:
        """Fallback search methods for better global coverage"""
        fallback_results = []
        
        # You could implement additional search APIs here
        # For now, return empty list as placeholder
        logger.info("Using fallback search methods")
        return fallback_results
    
    def scrape_content_enhanced(self, url: str, timeout: int = 15) -> str:
        """Enhanced content scraping optimized for AWS EC2 and global deployment"""
        try:
            logger.info(f"Scraping content from: {url}")
            
            # Multiple retry attempts with different strategies
            for attempt in range(3):
                try:
                    # Use session with appropriate timeout
                    response = self.session.get(
                        url, 
                        timeout=timeout,
                        allow_redirects=True,
                        stream=False
                    )
                    response.raise_for_status()
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if 'html' not in content_type:
                        logger.warning(f"Non-HTML content detected: {content_type}")
                        return f"Non-HTML content from {url}"
                    
                    # Parse with BeautifulSoup
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Remove unwanted elements more aggressively
                    unwanted_tags = ['script', 'style', 'nav', 'footer', 'header', 'aside', 
                                   'advertisement', 'ads', 'popup', 'modal', 'cookie-banner',
                                   'social-share', 'related-articles']
                    
                    for tag in unwanted_tags:
                        for element in soup.find_all(tag):
                            element.decompose()
                        for element in soup.find_all(attrs={"class": re.compile(tag, re.I)}):
                            element.decompose()
                        for element in soup.find_all(attrs={"id": re.compile(tag, re.I)}):
                            element.decompose()
                    
                    # Focus on main content with multiple strategies
                    main_content = self._extract_main_content(soup)
                    
                    if main_content:
                        text = main_content.get_text(separator=' ', strip=True)
                    else:
                        # Fallback to body content
                        body = soup.find('body')
                        text = body.get_text(separator=' ', strip=True) if body else soup.get_text(separator=' ', strip=True)
                    
                    # Clean and optimize text
                    text = self._clean_extracted_text(text)
                    
                    logger.info(f"Successfully scraped {len(text)} characters from {url}")
                    return text
                    
                except requests.RequestException as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                    if attempt == 2:  # Last attempt
                        raise
                    time.sleep(1)  # Brief wait before retry
                    
        except Exception as e:
            logger.error(f"Enhanced scraping failed for {url}: {e}")
            return f"Error scraping {url}: {str(e)}"
    
    def _extract_main_content(self, soup: BeautifulSoup) -> Optional:
        """Extract main content using multiple strategies"""
        # Strategy 1: Look for semantic HTML5 tags
        for tag in ['main', 'article']:
            main = soup.find(tag)
            if main:
                return main
        
        # Strategy 2: Look for content-related classes/IDs
        content_selectors = [
            'content', 'main-content', 'article-content', 'post-content',
            'entry-content', 'page-content', 'body-content', 'story-content'
        ]
        
        for selector in content_selectors:
            # Try class first
            main = soup.find('div', class_=re.compile(selector, re.I))
            if main:
                return main
            # Try ID
            main = soup.find('div', id=re.compile(selector, re.I))
            if main:
                return main
        
        # Strategy 3: Find the div with most text content
        divs = soup.find_all('div')
        if divs:
            content_div = max(divs, key=lambda d: len(d.get_text(strip=True)))
            if len(content_div.get_text(strip=True)) > 100:
                return content_div
        
        return None
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and optimize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common web artifacts
        text = re.sub(r'Cookie Policy.*?Accept', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Subscribe.*?Newsletter', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Share.*?Social', '', text, flags=re.IGNORECASE)
        
        # Remove excessive punctuation
        text = re.sub(r'[•\-]{3,}', ' ', text)
        
        # Limit size for processing efficiency
        max_length = 20000
        if len(text) > max_length:
            text = text[:max_length] + "... [Content truncated for processing]"
        
        return text.strip()
    
    def extract_financial_metrics(self, text: str) -> Dict[str, Any]:
        """Enhanced financial data extraction with global currency support"""
        metrics = {}
        
        # Financial patterns with currency support
        patterns = {
            "revenue": r'(?:revenue|sales|turnover)[s]?\s*(?:of|:)?\s*(?:[$€£¥]|USD|EUR|GBP|JPY|INR)?\s*([\d,.]+ ?(?:billion|million|thousand|B|M|K|bn|mn|cr|crore|lakh))',
            "profit": r'(?:net )?profit[s]?\s*(?:of|:)?\s*(?:[$€£¥]|USD|EUR|GBP|JPY|INR)?\s*([\d,.]+ ?(?:billion|million|thousand|B|M|K|bn|mn|cr|crore|lakh))',
            "market_cap": r'market cap(?:italization)?\s*(?:of|:)?\s*(?:[$€£¥]|USD|EUR|GBP|JPY|INR)?\s*([\d,.]+ ?(?:billion|million|thousand|B|M|K|bn|mn|cr|crore|lakh))',
            "valuation": r'valu(?:ed|ation)\s*(?:at|of)?\s*(?:[$€£¥]|USD|EUR|GBP|JPY|INR)?\s*([\d,.]+ ?(?:billion|million|thousand|B|M|K|bn|mn|cr|crore|lakh))',
            "employees": r'employ(?:s|ees|ment)\s*(?:of|:)?\s*([\d,]+)',
            "growth": r'growth\s*(?:rate|of)?\s*(?:of|:)?\s*([\d.]+%)',
            "margin": r'(?:profit|operating|gross)\s*margin\s*(?:of|:)?\s*([\d.]+%)'
        }
        
        for metric_name, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metrics[metric_name] = matches[0] if isinstance(matches[0], str) else matches[0]
        
        return metrics
    
    def analyze_market_sentiment(self, text: str) -> Dict[str, Any]:
        """Enhanced sentiment analysis for global markets"""
        positive_indicators = [
            "growth", "increase", "rise", "gain", "profit", "success", "strong", 
            "positive", "bullish", "optimistic", "expansion", "opportunity",
            "outperform", "beat expectations", "record high", "milestone"
        ]
        
        negative_indicators = [
            "decline", "decrease", "fall", "drop", "loss", "weak", "negative",
            "bearish", "pessimistic", "downturn", "challenge", "risk", "concern",
            "underperform", "miss expectations", "low", "struggle", "difficulty"
        ]
        
        neutral_indicators = [
            "stable", "flat", "unchanged", "steady", "maintained", "consistent",
            "expect", "forecast", "anticipate", "monitor", "watch", "track"
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in text_lower)
        neutral_count = sum(1 for indicator in neutral_indicators if indicator in text_lower)
        
        total_indicators = positive_count + negative_count + neutral_count
        
        if total_indicators == 0:
            return {"sentiment": "Unknown", "confidence": 0.0, "score": 0.0}
        
        sentiment_score = (positive_count - negative_count) / total_indicators
        confidence = total_indicators / max(len(text.split()) / 100, 1)
        
        if sentiment_score > 0.2:
            sentiment = "Positive"
        elif sentiment_score < -0.2:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return {
            "sentiment": sentiment,
            "score": sentiment_score,
            "confidence": min(confidence, 1.0),
            "positive_signals": positive_count,
            "negative_signals": negative_count,
            "neutral_signals": neutral_count
        }

# Simple LLM interface for DeepSeek and Ollama
class SimpleLLMEngine:
    """Simplified LLM engine for free models"""
    
    def __init__(self):
        self.models = FREE_MODEL_CONFIG
        self.current_backend = "ollama"  # Start with free local option
        
    def generate_analysis(self, query: str, context: str, analysis_type: str) -> str:
        """Generate business analysis using available free models"""
        try:
            # Create comprehensive prompt
            prompt = self._create_analysis_prompt(query, context, analysis_type)
            
            # Try different backends
            for backend in ["ollama", "deepseek"]:  # Prefer free local first
                try:
                    response = self._call_llm(backend, prompt)
                    if response:
                        return response
                except Exception as e:
                    logger.warning(f"Backend {backend} failed: {e}")
                    continue
            
            # Fallback to structured analysis without LLM
            return self._fallback_analysis(query, context, analysis_type)
            
        except Exception as e:
            logger.error(f"Analysis generation failed: {e}")
            return self._fallback_analysis(query, context, analysis_type)
    
    def _create_analysis_prompt(self, query: str, context: str, analysis_type: str) -> str:
        """Create comprehensive analysis prompt"""
        return f"""
As a business intelligence expert, analyze the following query using the provided context:

QUERY: {query}
ANALYSIS TYPE: {analysis_type}

CONTEXT DATA:
{context[:5000]}  # Limit context to avoid token limits

Please provide a comprehensive analysis including:
1. Executive Summary
2. Key Findings
3. Market Insights
4. Financial Metrics (if available)
5. Recommendations
6. Risk Assessment

Focus on actionable insights and data-driven conclusions.
"""
    
    def _call_llm(self, backend: str, prompt: str) -> Optional[str]:
        """Call specific LLM backend"""
        if backend == "ollama":
            return self._call_ollama(prompt)
        elif backend == "deepseek":
            return self._call_deepseek(prompt)
        return None
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama local model"""
        try:
            import json
            
            response = requests.post(
                f"{FREE_MODEL_CONFIG['ollama']['base_url']}/api/generate",
                json={
                    "model": FREE_MODEL_CONFIG['ollama']['models']['chat'],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.6,
                        "num_predict": 2048
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
        
        return None
    
    def _call_deepseek(self, prompt: str) -> Optional[str]:
        """Call DeepSeek API (requires API key)"""
        # This would require a valid API key
        # For now, return None to use fallback
        return None
    
    def _fallback_analysis(self, query: str, context: str, analysis_type: str) -> str:
        """Generate structured analysis without LLM"""
        return f"""
# {analysis_type} Analysis

## Executive Summary
Based on the available data sources, this analysis examines: {query}

## Key Findings
- Analysis covers multiple data points from recent sources
- Market data and business information have been aggregated
- Insights derived from credible business sources

## Data Summary
The analysis incorporated data from various sources including business news, 
financial reports, and market research. Key metrics and trends have been 
identified from the source material.

## Methodology
1. Web search across multiple business sources
2. Content extraction and analysis
3. Financial metric identification
4. Sentiment analysis of market conditions

## Recommendations
Based on the available data:
- Monitor key performance indicators
- Consider market trends and competitive landscape
- Evaluate financial metrics and growth patterns
- Assess risks and opportunities

**Note: This analysis was generated using data extraction and pattern matching. 
For more detailed insights, consider using AI-powered analysis with proper API keys.**
"""

# Initialize engines
@st.cache_resource
def get_web_engine():
    return GlobalWebScrapingEngine()

@st.cache_resource  
def get_llm_engine():
    return SimpleLLMEngine()

# Page configuration
st.set_page_config(
    page_title="🚀 Business Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }
    .analysis-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        border-top: 4px solid #667eea;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 0.8rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .status-healthy { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None

# Initialize engines
web_engine = get_web_engine()
llm_engine = get_llm_engine()

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 Business Intelligence Platform</h1>
    <p><strong>Optimized • Global Ready • Multi-LLM • Free Models Supported</strong></p>
    <p>AI-Powered Analysis • Enhanced Web Scraping • DeepSeek & Ollama Integration</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Model selection
    st.subheader("🤖 AI Backend")
    available_backends = ["ollama", "groq"]
    if CORE_AVAILABLE:
        available_backends.append("deepseek")
    
    selected_backend = st.selectbox(
        "Primary Model",
        available_backends,
        help="Ollama: Free local models • Groq: Fast inference • DeepSeek: Advanced reasoning"
    )
    
    # Analysis type
    st.subheader("📊 Analysis Type")
    analysis_types = [
        "🏢 Market Research",
        "⚔️ Competitive Analysis", 
        "💰 Investment Analysis",
        "🔍 Company Intelligence",
        "📈 Trend Analysis",
        "💹 Financial Analysis",
        "🌐 Industry Overview",
        "❓ Custom Analysis"
    ]
    
    selected_analysis = st.selectbox("Analysis Focus", analysis_types)
    analysis_type = selected_analysis.split(" ", 1)[1]  # Remove emoji
    
    # Search settings
    st.subheader("🌐 Search Settings")
    max_sources = st.slider("Max Sources", 3, 15, 8)
    search_region = st.selectbox(
        "Search Region",
        ["us-en", "gb-en", "au-en", "ca-en", "in-en", "de-de", "fr-fr", "jp-jp"],
        help="Optimize search for specific regions"
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Settings"):
        deep_analysis = st.checkbox("🔍 Deep Content Analysis", value=True)
        sentiment_analysis = st.checkbox("📊 Sentiment Analysis", value=True)
        financial_extraction = st.checkbox("💰 Financial Data Extraction", value=True)
        enhanced_scraping = st.checkbox("🌐 Enhanced Global Scraping", value=True)
    
    # Session statistics
    st.markdown("---")
    st.subheader("📊 Session Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Analyses", len(st.session_state.analysis_history))
        st.metric("Backend", selected_backend.upper())
    with col2:
        total_sources = sum(len(a.get("sources", [])) for a in st.session_state.analysis_history)
        st.metric("Sources", total_sources)
        st.metric("Region", search_region.upper())

# Main interface
col1, col2 = st.columns([3, 1])

with col1:
    st.header("🔍 Business Analysis Center")
    
    # Query input with templates
    if analysis_type != "Custom Analysis":
        templates = {
            "Market Research": "Analyze the market size, growth trends, and opportunities for [INDUSTRY/PRODUCT]. Include competitive landscape and future outlook.",
            "Competitive Analysis": "Compare [COMPANY] with main competitors. Analyze market positioning, strengths, weaknesses, and strategic advantages.",
            "Investment Analysis": "Evaluate [COMPANY/SECTOR] as investment opportunity. Include financial metrics, growth potential, risks, and investment thesis.",
            "Company Intelligence": "Provide comprehensive analysis of [COMPANY] including business model, recent developments, financial performance, and strategic direction.",
            "Trend Analysis": "Analyze current and emerging trends in [INDUSTRY/TECHNOLOGY]. Include adoption rates, market impact, and future predictions.",
            "Financial Analysis": "Conduct detailed financial analysis of [COMPANY] including revenue, profitability, debt analysis, and financial health assessment.",
            "Industry Overview": "Provide comprehensive overview of [INDUSTRY] including market dynamics, key players, trends, and growth opportunities."
        }
        
        query = st.text_area(
            f"📝 {analysis_type} Query:",
            value=templates.get(analysis_type, ""),
            height=120,
            help="Replace [PLACEHOLDERS] with specific companies, industries, or topics"
        )
    else:
        query = st.text_area(
            "📝 Enter your business question:",
            placeholder="e.g., What are the growth prospects for renewable energy companies in emerging markets?",
            height=120
        )

with col2:
    st.markdown("### 🚀 Actions")
    
    # Main analysis button
    if st.button("🔍 **Start Analysis**", disabled=not query.strip()):
        if query.strip():
            st.session_state.current_analysis = {
                "query": query,
                "type": analysis_type,
                "backend": selected_backend,
                "timestamp": datetime.now(),
                "status": "running",
                "config": {
                    "max_sources": max_sources,
                    "region": search_region,
                    "deep_analysis": deep_analysis,
                    "sentiment_analysis": sentiment_analysis,
                    "financial_extraction": financial_extraction,
                    "enhanced_scraping": enhanced_scraping
                }
            }
            st.rerun()
    
    # Quick examples
    if st.button("💡 **Show Examples**"):
        examples = [
            "Tesla vs Chinese EV manufacturers competitive analysis",
            "Microsoft Azure market position in cloud computing",
            "Renewable energy investment trends in Asia 2024",
            "Apple services revenue growth analysis",
            "Cryptocurrency market sentiment analysis"
        ]
        for example in examples:
            st.info(f"💡 {example}")
    
    if st.button("🗑️ **Clear History**"):
        st.session_state.analysis_history = []
        st.session_state.current_analysis = None
        st.rerun()

# Analysis execution
if st.session_state.current_analysis and st.session_state.current_analysis["status"] == "running":
    analysis = st.session_state.current_analysis
    config = analysis["config"]
    
    st.markdown(f"""
    <div class="analysis-container">
        <h3>🔬 {analysis['type']} Analysis</h3>
        <p><strong>Query:</strong> {analysis['query']}</p>
        <p><strong>Backend:</strong> {analysis['backend'].upper()} • <strong>Region:</strong> {config['region'].upper()} • <strong>Sources:</strong> {config['max_sources']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress tracking
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if st.button("⏹️ Stop Analysis"):
            st.session_state.current_analysis["status"] = "stopped"
            st.rerun()
    
    try:
        # Phase 1: Web Search
        status_text.text("🌐 Searching global sources...")
        progress_bar.progress(0.2)
        
        search_results = web_engine.search_web_enhanced(
            analysis["query"], 
            config["max_sources"], 
            config["region"]
        )
        
        # Phase 2: Content Scraping  
        status_text.text("📄 Extracting content...")
        progress_bar.progress(0.4)
        
        detailed_sources = []
        for i, result in enumerate(search_results):
            if config["enhanced_scraping"]:
                content = web_engine.scrape_content_enhanced(result["url"])
                result["content"] = content
            detailed_sources.append(result)
            
            progress_bar.progress(0.4 + (i / len(search_results)) * 0.3)
        
        # Phase 3: Data Analysis
        status_text.text("📊 Analyzing data...")
        progress_bar.progress(0.7)
        
        # Aggregate content
        all_content = " ".join([
            f"{source.get('title', '')} {source.get('snippet', '')} {source.get('content', '')}"
            for source in detailed_sources
        ])
        
        # Extract insights
        financial_data = {}
        sentiment_data = {}
        
        if config["financial_extraction"]:
            financial_data = web_engine.extract_financial_metrics(all_content)
        
        if config["sentiment_analysis"]:
            sentiment_data = web_engine.analyze_market_sentiment(all_content)
        
        # Phase 4: AI Analysis
        status_text.text("🧠 Generating AI analysis...")
        progress_bar.progress(0.9)
        
        ai_analysis = ""
        if config["deep_analysis"]:
            ai_analysis = llm_engine.generate_analysis(
                analysis["query"],
                all_content[:8000],  # Limit for processing
                analysis["type"]
            )
        
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")
        
        # Store results
        analysis_results = {
            "sources": detailed_sources,
            "financial_data": financial_data,
            "sentiment_data": sentiment_data,
            "ai_analysis": ai_analysis,
            "summary": f"Comprehensive {analysis['type'].lower()} completed with {len(detailed_sources)} sources analyzed."
        }
        
        analysis["results"] = analysis_results
        analysis["status"] = "completed"
        analysis["completion_time"] = datetime.now()
        st.session_state.analysis_history.append(analysis)
        
        # Display results
        with results_container:
            st.markdown("## 📋 Analysis Results")
            
            # Key metrics
            st.markdown("### 📊 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Sources Analyzed", len(detailed_sources))
            with col2:
                avg_relevance = sum(s['relevance_score'] for s in detailed_sources) / len(detailed_sources)
                st.metric("Avg Relevance", f"{avg_relevance:.1f}/10")
            with col3:
                if sentiment_data:
                    st.metric("Market Sentiment", sentiment_data.get("sentiment", "Unknown"))
                else:
                    st.metric("Backend", analysis["backend"].upper())
            with col4:
                if financial_data:
                    st.metric("Financial Metrics", len(financial_data))
                else:
                    st.metric("Region", config["region"].upper())
            
            # AI Analysis
            if ai_analysis:
                st.markdown("### 🧠 AI Analysis")
                st.markdown(ai_analysis)
            
            # Financial Data
            if financial_data:
                st.markdown("### 💰 Financial Data")
                fin_cols = st.columns(min(len(financial_data), 3))
                for i, (metric, value) in enumerate(financial_data.items()):
                    with fin_cols[i % 3]:
                        st.metric(metric.replace("_", " ").title(), value)
            
            # Sentiment Analysis
            if sentiment_data:
                st.markdown("### 📊 Sentiment Analysis")
                sent_col1, sent_col2, sent_col3 = st.columns(3)
                with sent_col1:
                    st.metric("Overall Sentiment", sentiment_data["sentiment"])
                with sent_col2:
                    st.metric("Confidence", f"{sentiment_data.get('confidence', 0):.1%}")
                with sent_col3:
                    st.metric("Score", f"{sentiment_data.get('score', 0):.2f}")
            
            # Top Sources
            st.markdown("### 📚 Data Sources")
            for i, source in enumerate(detailed_sources[:5]):
                with st.expander(f"📄 Source {i+1}: {source['title'][:60]}..."):
                    st.write(f"**URL:** {source['url']}")
                    st.write(f"**Relevance:** {source['relevance_score']:.1f}/10")
                    st.write(f"**Snippet:** {source['snippet'][:200]}...")
                    if source.get("content") and config["enhanced_scraping"]:
                        content_preview = source['content'][:400] + "..." if len(source['content']) > 400 else source['content']
                        st.write(f"**Content Preview:** {content_preview}")
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        logger.error(f"Analysis execution failed: {e}")
        st.session_state.current_analysis["status"] = "failed"

# Analysis History
if st.session_state.analysis_history:
    st.markdown("---")
    st.header("📈 Analysis History")
    
    for i, hist_analysis in enumerate(reversed(st.session_state.analysis_history[-3:])):
        with st.expander(f"📊 {hist_analysis['type']} - {hist_analysis['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
            st.write(f"**Query:** {hist_analysis['query'][:100]}...")
            st.write(f"**Status:** {hist_analysis['status']}")
            if hist_analysis.get('results'):
                results = hist_analysis['results']
                st.write(f"**Sources:** {len(results.get('sources', []))}")
                if results.get('sentiment_data'):
                    st.write(f"**Sentiment:** {results['sentiment_data'].get('sentiment', 'Unknown')}")
                if results.get('financial_data'):
                    st.write(f"**Financial Metrics:** {len(results['financial_data'])}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>🚀 Business Intelligence Platform</strong> - Optimized for Global Deployment</p>
    <p>Enhanced Web Scraping • Multi-LLM Support • Free Models • AWS EC2 Ready</p>
</div>
""", unsafe_allow_html=True)