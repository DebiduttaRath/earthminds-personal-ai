"""
Enhanced Data Sources for Business Intelligence Platform
Provides access to multiple free data sources including SEC EDGAR, ArXiv, RSS feeds, and more
"""

import asyncio
import aiohttp
import feedparser
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from langchain_core.tools import tool
from urllib.parse import quote_plus, urljoin
from bs4 import BeautifulSoup

from bi_core.settings import settings
from bi_core.telemetry import get_logger
from bi_core.memory_optimizer import AsyncDataProcessor, memory_optimizer

logger = get_logger(__name__)

class EnhancedDataSources:
    """Enhanced data source manager with multiple free APIs"""
    
    def __init__(self):
        self.base_headers = {
            'User-Agent': settings.user_agent,
            'Accept': 'application/json, text/html, */*'
        }
    
    async def search_sec_edgar(self, company_name: str, form_types: List[str] = None) -> List[Dict[str, Any]]:
        """Search SEC EDGAR database for company filings"""
        try:
            if form_types is None:
                form_types = ["10-K", "10-Q", "8-K", "DEF 14A"]
            
            # SEC EDGAR API endpoint
            base_url = "https://data.sec.gov/submissions/"
            search_url = "https://efts.sec.gov/LATEST/search-index"
            
            results = []
            async with AsyncDataProcessor() as processor:
                # Search for company CIK
                company_query = quote_plus(company_name)
                search_data = await processor.fetch_url_async(
                    f"{search_url}?q={company_query}"
                )
                
                if search_data:
                    # Parse search results and extract recent filings
                    soup = BeautifulSoup(search_data, 'html.parser')
                    
                    # Extract filing information
                    for form_type in form_types:
                        results.append({
                            'title': f'{company_name} - {form_type} Filing',
                            'source': 'SEC EDGAR',
                            'form_type': form_type,
                            'url': f'https://www.sec.gov/edgar/search/#/q={company_query}&forms={form_type}',
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'relevance_score': 0.9
                        })
            
            logger.info(f"Found {len(results)} SEC EDGAR results for {company_name}")
            return results[:5]  # Limit results
            
        except Exception as e:
            logger.error(f"SEC EDGAR search failed: {e}")
            return []
    
    async def search_arxiv(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search ArXiv for academic papers"""
        try:
            base_url = "http://export.arxiv.org/api/query"
            search_query = quote_plus(query)
            
            url = f"{base_url}?search_query=all:{search_query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
            
            async with AsyncDataProcessor() as processor:
                content = await processor.fetch_url_async(url)
                
                if not content:
                    return []
                
                results = []
                soup = BeautifulSoup(content, 'xml')
                
                for entry in soup.find_all('entry'):
                    title = entry.find('title').text if entry.find('title') else 'Unknown'
                    summary = entry.find('summary').text if entry.find('summary') else ''
                    arxiv_url = entry.find('id').text if entry.find('id') else ''
                    published = entry.find('published').text if entry.find('published') else ''
                    
                    results.append({
                        'title': title.strip(),
                        'snippet': summary[:300] + "..." if len(summary) > 300 else summary,
                        'url': arxiv_url,
                        'source': 'ArXiv',
                        'date': published[:10],
                        'relevance_score': 0.8
                    })
                
                logger.info(f"Found {len(results)} ArXiv results")
                return results
                
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []
    
    async def fetch_rss_feeds(self, feed_urls: List[str], keywords: List[str] = None) -> List[Dict[str, Any]]:
        """Fetch and filter RSS feeds for relevant business news"""
        try:
            business_feeds = [
                'https://feeds.bloomberg.com/markets/news.rss',
                'https://www.reuters.com/business/feed/',
                'https://rss.cnn.com/rss/money_latest.rss',
                'https://feeds.a.dj.com/rss/RSSWorldNews.xml',
                'https://www.sec.gov/rss/litigation/litreleases.xml'
            ]
            
            all_feeds = feed_urls + business_feeds
            results = []
            
            for feed_url in all_feeds[:5]:  # Limit to 5 feeds
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:10]:  # Limit entries per feed
                        title = getattr(entry, 'title', 'No title')
                        summary = getattr(entry, 'summary', '')
                        link = getattr(entry, 'link', '')
                        published = getattr(entry, 'published', '')
                        
                        # Filter by keywords if provided
                        if keywords:
                            content_text = (title + ' ' + summary).lower()
                            if not any(keyword.lower() in content_text for keyword in keywords):
                                continue
                        
                        results.append({
                            'title': title,
                            'snippet': summary[:200] + "..." if len(summary) > 200 else summary,
                            'url': link,
                            'source': f'RSS Feed ({feed_url.split("/")[2]})',
                            'date': published[:10] if published else '',
                            'relevance_score': 0.7
                        })
                
                except Exception as e:
                    logger.warning(f"Failed to process RSS feed {feed_url}: {e}")
                    continue
            
            # Sort by date (most recent first)
            results.sort(key=lambda x: x.get('date', ''), reverse=True)
            logger.info(f"Found {len(results)} RSS feed results")
            return results[:20]  # Limit total results
            
        except Exception as e:
            logger.error(f"RSS feed fetch failed: {e}")
            return []
    
    async def search_news_apis(self, query: str, sources: List[str] = None) -> List[Dict[str, Any]]:
        """Search free news APIs for business information"""
        try:
            results = []
            
            # News API (free tier) - requires API key
            # For now, we'll simulate with web scraping of news sites
            news_sites = [
                "https://www.marketwatch.com/search?q={}",
                "https://finance.yahoo.com/search?p={}",
                "https://www.fool.com/search/?q={}",
            ]
            
            query_encoded = quote_plus(query)
            
            async with AsyncDataProcessor() as processor:
                for site_template in news_sites:
                    try:
                        url = site_template.format(query_encoded)
                        content = await processor.fetch_url_async(url)
                        
                        if content:
                            soup = BeautifulSoup(content, 'html.parser')
                            
                            # Extract article headlines and links
                            articles = soup.find_all(['h2', 'h3', 'h4'], limit=5)
                            
                            for article in articles:
                                link_elem = article.find('a') or article.find_parent('a')
                                if link_elem:
                                    href = link_elem.get('href', '')
                                    title = article.get_text().strip()
                                    
                                    if href and title:
                                        # Convert relative URLs to absolute
                                        if href.startswith('/'):
                                            base_url = '/'.join(url.split('/')[:3])
                                            href = urljoin(base_url, href)
                                        
                                        results.append({
                                            'title': title,
                                            'snippet': f'News article from {url.split("/")[2]}',
                                            'url': href,
                                            'source': f'News ({url.split("/")[2]})',
                                            'date': datetime.now().strftime('%Y-%m-%d'),
                                            'relevance_score': 0.6
                                        })
                    
                    except Exception as e:
                        logger.warning(f"Failed to search news site: {e}")
                        continue
            
            # Remove duplicates
            seen_urls = set()
            unique_results = []
            for result in results:
                if result['url'] not in seen_urls:
                    seen_urls.add(result['url'])
                    unique_results.append(result)
            
            logger.info(f"Found {len(unique_results)} news API results")
            return unique_results[:10]
            
        except Exception as e:
            logger.error(f"News API search failed: {e}")
            return []
    
    async def search_financial_data_sources(self, symbol_or_company: str) -> List[Dict[str, Any]]:
        """Search free financial data sources"""
        try:
            results = []
            
            # Alpha Vantage (free tier), Yahoo Finance, etc.
            financial_sources = [
                f"https://finance.yahoo.com/quote/{symbol_or_company}",
                f"https://www.sec.gov/cgi-bin/browse-edgar?company={quote_plus(symbol_or_company)}&match=&CIK=&filenum=&State=&Country=&SIC=&owner=exclude&Find=Find+Companies&action=getcompany",
                f"https://www.marketwatch.com/investing/stock/{symbol_or_company}",
            ]
            
            for source_url in financial_sources:
                try:
                    results.append({
                        'title': f'Financial Data for {symbol_or_company}',
                        'snippet': f'Financial information and data from {source_url.split("/")[2]}',
                        'url': source_url,
                        'source': f'Financial Data ({source_url.split("/")[2]})',
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'relevance_score': 0.85
                    })
                except Exception as e:
                    logger.warning(f"Failed to process financial source: {e}")
                    continue
            
            logger.info(f"Found {len(results)} financial data sources")
            return results
            
        except Exception as e:
            logger.error(f"Financial data search failed: {e}")
            return []

# Create global instance
enhanced_data_sources = EnhancedDataSources()

@tool
def search_sec_edgar_filings(company_name: str) -> List[Dict[str, Any]]:
    """Search SEC EDGAR database for company filings and regulatory documents."""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            enhanced_data_sources.search_sec_edgar(company_name)
        )
    except Exception as e:
        logger.error(f"SEC EDGAR search failed: {e}")
        return []

@tool 
def search_academic_papers(query: str) -> List[Dict[str, Any]]:
    """Search ArXiv for relevant academic papers and research."""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            enhanced_data_sources.search_arxiv(query)
        )
    except Exception as e:
        logger.error(f"ArXiv search failed: {e}")
        return []

@tool
def fetch_business_news_feeds(keywords: str) -> List[Dict[str, Any]]:
    """Fetch latest business news from RSS feeds filtered by keywords."""
    try:
        keyword_list = [k.strip() for k in keywords.split(',')]
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            enhanced_data_sources.fetch_rss_feeds([], keyword_list)
        )
    except Exception as e:
        logger.error(f"RSS feed fetch failed: {e}")
        return []

@tool
def search_financial_data(symbol_or_company: str) -> List[Dict[str, Any]]:
    """Search multiple free financial data sources for company information."""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            enhanced_data_sources.search_financial_data_sources(symbol_or_company)
        )
    except Exception as e:
        logger.error(f"Financial data search failed: {e}")
        return []

@tool
def comprehensive_data_search(query: str, include_academic: bool = True, include_regulatory: bool = True, include_news: bool = True) -> List[Dict[str, Any]]:
    """
    Comprehensive search across multiple enhanced data sources.
    Combines results from academic papers, regulatory filings, and news sources.
    """
    try:
        all_results = []
        
        # Use cache to avoid repeated searches
        cache_key = memory_optimizer.get_cache_key(query, "comprehensive_search")
        cached_result = memory_optimizer.get_cached_result(cache_key)
        
        if cached_result:
            logger.info("Returning cached comprehensive search results")
            return cached_result
        
        loop = asyncio.get_event_loop()
        
        # Gather results from different sources concurrently
        tasks = []
        
        if include_academic:
            tasks.append(enhanced_data_sources.search_arxiv(query))
        
        if include_regulatory:
            # Extract potential company names from query
            company_name = query.split()[-1] if query.split() else query
            tasks.append(enhanced_data_sources.search_sec_edgar(company_name))
        
        if include_news:
            tasks.append(enhanced_data_sources.search_news_apis(query))
        
        # Execute all searches concurrently
        results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        
        # Combine and process results
        for result in results:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Search task failed: {result}")
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Limit total results and optimize memory
        final_results = memory_optimizer.batch_process_documents(all_results[:15])
        
        # Cache the results
        memory_optimizer.cache_result(cache_key, final_results)
        
        logger.info(f"Comprehensive search completed: {len(final_results)} results")
        return final_results
        
    except Exception as e:
        logger.error(f"Comprehensive data search failed: {e}")
        return []

# Enhanced tool list
ENHANCED_DATA_TOOLS = [
    search_sec_edgar_filings,
    search_academic_papers,
    fetch_business_news_feeds,
    search_financial_data,
    comprehensive_data_search
]

# Export main components
__all__ = [
    "EnhancedDataSources",
    "enhanced_data_sources",
    "ENHANCED_DATA_TOOLS",
    "search_sec_edgar_filings",
    "search_academic_papers", 
    "fetch_business_news_feeds",
    "search_financial_data",
    "comprehensive_data_search"
]