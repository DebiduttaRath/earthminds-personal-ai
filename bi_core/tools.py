"""
Business Intelligence Tools
Enhanced search and analysis tools for comprehensive business intelligence
"""

import requests
import json
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.tools import tool
from bi_core.settings import settings
from bi_core.telemetry import get_logger
from utils.web_scraper import get_website_text_content
import re
from datetime import datetime, timedelta

logger = get_logger(__name__)

# Configure requests session with business-appropriate headers
session = requests.Session()
session.headers.update({
    "User-Agent": settings.user_agent,
    "Accept": "application/json, text/html, application/xhtml+xml, application/xml;q=0.9, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
})

class SearchResult(TypedDict):
    title: str
    snippet: str
    url: str
    relevance_score: float
    date: Optional[str]

class CompanyInfo(TypedDict):
    name: str
    industry: str
    market_cap: Optional[str]
    revenue: Optional[str]
    employees: Optional[str]
    headquarters: Optional[str]
    website: Optional[str]

class FinancialMetric(TypedDict):
    metric: str
    value: str
    period: str
    currency: Optional[str]

@retry(stop=stop_after_attempt(settings.http_retries),
       wait=wait_exponential(multiplier=1, min=0.5, max=10),
       retry=retry_if_exception_type((requests.RequestException, ConnectionError)))
def _safe_get(url: str, **kwargs) -> requests.Response:
    """Make a safe HTTP GET request with retries and error handling"""
    try:
        response = session.get(url, timeout=settings.http_timeout, **kwargs)
        response.raise_for_status()
        return response
    except Exception as e:
        logger.error(f"HTTP request failed for {url}: {e}")
        raise

@tool
def business_wiki_search(query: str) -> List[SearchResult]:
    """
    Search Wikipedia for business and company information.
    Optimized for business intelligence queries about companies, industries, and markets.
    """
    try:
        logger.info(f"Business Wikipedia search: {query}")
        
        # Enhance query for business searches
        business_terms = ["company", "corporation", "industry", "market", "business"]
        if not any(term in query.lower() for term in business_terms):
            query = f"{query} company business"
        
        params = {
            "action": "query",
            "list": "search",
            "format": "json",
            "srsearch": query,
            "srlimit": 8,  # More results for business intelligence
            "srprop": "snippet|titlesnippet|size|timestamp",
            "utf8": 1,
            "origin": "*"
        }
        
        response = _safe_get("https://en.wikipedia.org/w/api.php", params=params)
        data = response.json()
        
        results: List[SearchResult] = []
        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "")
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            
            # Clean snippet
            snippet = BeautifulSoup(item.get("snippet", ""), "html.parser").get_text()
            snippet = re.sub(r'\s+', ' ', snippet).strip()
            
            # Calculate relevance based on business terms
            relevance_score = sum(1 for term in business_terms if term in snippet.lower()) / len(business_terms)
            
            results.append({
                "title": title,
                "snippet": snippet,
                "url": url,
                "relevance_score": relevance_score,
                "date": item.get("timestamp", "").split("T")[0] if item.get("timestamp") else None
            })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        if not results:
            results = [{"title": "No results found", "snippet": "", "url": "", "relevance_score": 0.0, "date": None}]
            
        logger.info(f"Found {len(results)} Wikipedia results")
        return results[:5]  # Return top 5 most relevant
        
    except Exception as e:
        logger.error(f"Wikipedia search failed: {e}")
        return [{"title": "Search failed", "snippet": f"Error: {str(e)}", "url": "", "relevance_score": 0.0, "date": None}]

@tool
def business_web_search(query: str) -> List[SearchResult]:
    """
    Enhanced DuckDuckGo search optimized for business intelligence.
    Focuses on recent business news, market data, and company information.
    """
    try:
        logger.info(f"Business web search: {query}")
        
        results: List[SearchResult] = []
        
        with DDGS() as ddgs:
            # Get more results for better analysis
            search_results = list(ddgs.text(
                query,
                max_results=10,
                region='us-en',  # US English for business focus
                timelimit='y'    # Last year for recent business info
            ))
            
            for result in search_results:
                title = result.get("title", "")
                snippet = result.get("body", "")
                url = result.get("href", "")
                
                # Calculate business relevance
                business_keywords = [
                    "company", "corporation", "business", "market", "industry", "revenue", 
                    "profit", "earnings", "financial", "investment", "analysis", "report"
                ]
                
                relevance_score = sum(1 for keyword in business_keywords 
                                    if keyword in (title + " " + snippet).lower()) / len(business_keywords)
                
                # Boost score for business domains
                business_domains = [
                    "bloomberg.com", "reuters.com", "wsj.com", "ft.com", "marketwatch.com",
                    "fool.com", "sec.gov", "investor.gov", "yahoo.com/finance", "google.com/finance"
                ]
                
                if any(domain in url for domain in business_domains):
                    relevance_score += 0.3
                
                results.append({
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                    "relevance_score": min(relevance_score, 1.0),
                    "date": None  # DuckDuckGo doesn't provide dates directly
                })
        
        # Sort by relevance and recency
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        if not results:
            results = [{"title": "No results found", "snippet": "", "url": "", "relevance_score": 0.0, "date": None}]
        
        logger.info(f"Found {len(results)} web search results")
        return results[:5]
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return [{"title": "Search failed", "snippet": f"Error: {str(e)}", "url": "", "relevance_score": 0.0, "date": None}]

@tool
def fetch_business_content(url: str) -> str:
    """
    Fetch and extract clean text content from business websites and reports.
    Optimized for financial reports, press releases, and business documents.
    """
    try:
        logger.info(f"Fetching business content from: {url}")
        
        # Use the web scraper utility
        content = get_website_text_content(url)
        
        if not content:
            # Fallback to basic scraping
            response = _safe_get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "advertisement"]):
                element.decompose()
            
            # Focus on main content areas
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|body'))
            
            if main_content:
                content = main_content.get_text(separator=' ', strip=True)
            else:
                content = soup.get_text(separator=' ', strip=True)
        
        # Clean up the content
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        # Limit content size for processing
        max_length = 25000
        if len(content) > max_length:
            content = content[:max_length] + "... [Content truncated]"
        
        logger.info(f"Extracted {len(content)} characters from {url}")
        return content
        
    except Exception as e:
        logger.error(f"Failed to fetch content from {url}: {e}")
        return f"Error fetching content from {url}: {str(e)}"

@tool
def analyze_financial_metrics(text: str) -> List[FinancialMetric]:
    """
    Extract financial metrics from text content.
    Identifies revenue, profit, market cap, and other key business metrics.
    """
    try:
        logger.info("Analyzing financial metrics from text")
        
        metrics: List[FinancialMetric] = []
        
        # Financial metric patterns
        patterns = {
            "revenue": r'revenue[s]?\s*(?:of|:)?\s*[\$]?([\d,.]+ (?:billion|million|thousand|B|M|K))',
            "profit": r'profit[s]?\s*(?:of|:)?\s*[\$]?([\d,.]+ (?:billion|million|thousand|B|M|K))',
            "earnings": r'earnings[s]?\s*(?:of|:)?\s*[\$]?([\d,.]+ (?:billion|million|thousand|B|M|K))',
            "market_cap": r'market cap(?:italization)?\s*(?:of|:)?\s*[\$]?([\d,.]+ (?:billion|million|thousand|B|M|K))',
            "employees": r'employ(?:s|ees)\s*(?:of|:)?\s*([\d,]+)',
            "growth": r'growth\s*(?:of|:)?\s*([\d.]+%)',
            "margin": r'margin[s]?\s*(?:of|:)?\s*([\d.]+%)'
        }
        
        for metric_name, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1)
                
                # Try to extract period context
                context = text[max(0, match.start()-100):match.end()+100]
                period_match = re.search(r'(Q[1-4]\s+20\d{2}|20\d{2}|fiscal\s+20\d{2})', context, re.IGNORECASE)
                period = period_match.group(1) if period_match else "Not specified"
                
                currency = "USD" if "$" in match.group(0) else None
                
                metrics.append({
                    "metric": metric_name.title(),
                    "value": value,
                    "period": period,
                    "currency": currency
                })
        
        # Remove duplicates
        seen = set()
        unique_metrics = []
        for metric in metrics:
            key = (metric["metric"], metric["value"], metric["period"])
            if key not in seen:
                seen.add(key)
                unique_metrics.append(metric)
        
        logger.info(f"Extracted {len(unique_metrics)} financial metrics")
        return unique_metrics[:10]  # Return top 10 metrics
        
    except Exception as e:
        logger.error(f"Financial metrics analysis failed: {e}")
        return []

@tool
def company_news_search(company_name: str, days_back: int = 30) -> List[SearchResult]:
    """
    Search for recent news about a specific company.
    Focuses on business developments, earnings, partnerships, etc.
    """
    try:
        logger.info(f"Searching news for company: {company_name}")
        
        # Construct news-focused search query
        news_terms = ["news", "earnings", "announced", "partnership", "acquisition", "financial"]
        query = f'"{company_name}" ({" OR ".join(news_terms)})'
        
        results: List[SearchResult] = []
        
        with DDGS() as ddgs:
            # Search for recent news
            search_results = list(ddgs.text(
                query,
                max_results=8,
                region='us-en',
                timelimit='m' if days_back <= 30 else 'y'  # Last month or year
            ))
            
            for result in search_results:
                title = result.get("title", "")
                snippet = result.get("body", "")
                url = result.get("href", "")
                
                # Calculate news relevance
                news_keywords = [
                    "announced", "reports", "earnings", "quarterly", "revenue", "profit",
                    "partnership", "acquisition", "merger", "investment", "launch", "contract"
                ]
                
                relevance_score = sum(1 for keyword in news_keywords 
                                    if keyword in (title + " " + snippet).lower()) / len(news_keywords)
                
                # Boost for news sources
                news_domains = [
                    "reuters.com", "bloomberg.com", "wsj.com", "cnbc.com", "marketwatch.com",
                    "businesswire.com", "prnewswire.com", "sec.gov"
                ]
                
                if any(domain in url for domain in news_domains):
                    relevance_score += 0.4
                
                # Check if company name is prominently mentioned
                if company_name.lower() in title.lower():
                    relevance_score += 0.3
                
                results.append({
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                    "relevance_score": min(relevance_score, 1.0),
                    "date": None
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        if not results:
            results = [{"title": "No recent news found", "snippet": "", "url": "", "relevance_score": 0.0, "date": None}]
        
        logger.info(f"Found {len(results)} news results for {company_name}")
        return results[:5]
        
    except Exception as e:
        logger.error(f"Company news search failed: {e}")
        return [{"title": "News search failed", "snippet": f"Error: {str(e)}", "url": "", "relevance_score": 0.0, "date": None}]

@tool
def business_calculator(expression: str) -> str:
    """
    Safe calculator for business and financial calculations.
    Supports basic math, percentages, and financial formulas.
    """
    try:
        logger.info(f"Business calculation: {expression}")
        
        # Import math for financial calculations
        import math
        
        # Safe evaluation environment
        allowed_names = {
            k: getattr(math, k) for k in dir(math) if not k.startswith("_")
        }
        allowed_names.update({
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len, "pow": pow
        })
        
        # Replace common financial terms
        expression = expression.replace("%", "/100")
        expression = re.sub(r'\$', '', expression)  # Remove dollar signs
        expression = re.sub(r'[,]', '', expression)  # Remove commas
        
        # Handle common financial calculations
        if "compound" in expression.lower() and "interest" in expression.lower():
            # Compound interest calculation hint
            return "For compound interest: A = P(1 + r/n)^(nt) where P=principal, r=rate, n=compounding frequency, t=time"
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        
        # Format financial results
        if isinstance(result, (int, float)):
            if result > 1000000:
                formatted_result = f"${result:,.0f} ({result/1000000:.2f}M)"
            elif result > 1000:
                formatted_result = f"${result:,.0f} ({result/1000:.1f}K)"
            else:
                formatted_result = f"${result:.2f}"
        else:
            formatted_result = str(result)
        
        logger.info(f"Calculation result: {formatted_result}")
        return formatted_result
        
    except Exception as e:
        logger.error(f"Business calculation failed: {e}")
        return f"Calculation error: {str(e)}. Please check your expression."

@tool
def market_data_search(ticker_or_company: str) -> Dict[str, Any]:
    """
    Search for basic market data and company information.
    Note: This is a simplified version - production systems would integrate with financial data APIs.
    """
    try:
        logger.info(f"Market data search for: {ticker_or_company}")
        
        # Search for company information
        query = f"{ticker_or_company} stock market cap revenue financial data"
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            
            market_info = {
                "company": ticker_or_company,
                "search_results": [],
                "extracted_data": {}
            }
            
            for result in results:
                title = result.get("title", "")
                snippet = result.get("body", "")
                url = result.get("href", "")
                
                # Look for financial data in snippets
                if any(term in snippet.lower() for term in ["market cap", "revenue", "stock price", "earnings"]):
                    market_info["search_results"].append({
                        "title": title,
                        "snippet": snippet,
                        "url": url
                    })
                    
                    # Try to extract basic metrics
                    metrics = analyze_financial_metrics(snippet)
                    if metrics:
                        for metric in metrics:
                            market_info["extracted_data"][metric["metric"]] = {
                                "value": metric["value"],
                                "period": metric["period"]
                            }
            
            logger.info(f"Found market data for {ticker_or_company}")
            return market_info
            
    except Exception as e:
        logger.error(f"Market data search failed: {e}")
        return {"error": str(e), "company": ticker_or_company}

# Tool registry for the graph
BUSINESS_TOOLS = [
    business_wiki_search,
    business_web_search,
    fetch_business_content,
    analyze_financial_metrics,
    company_news_search,
    business_calculator,
    market_data_search
]
