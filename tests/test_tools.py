"""
Test suite for Business Intelligence Tools
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from bi_core.tools import (
    business_wiki_search, business_web_search, fetch_business_content,
    analyze_financial_metrics, company_news_search, business_calculator,
    market_data_search
)

class TestBusinessTools:
    """Test business intelligence tools"""
    
    def test_business_calculator_basic_math(self):
        """Test basic mathematical operations"""
        result = business_calculator("100 + 50")
        assert "$150.00" in result
        
        result = business_calculator("1000000")
        assert "1.0M" in result
        
        result = business_calculator("1500000")
        assert "1.5M" in result
    
    def test_business_calculator_percentage(self):
        """Test percentage calculations"""
        result = business_calculator("100 * 25%")
        assert "25" in result
    
    def test_business_calculator_error_handling(self):
        """Test calculator error handling"""
        result = business_calculator("invalid expression")
        assert "error" in result.lower()
    
    def test_analyze_financial_metrics_basic(self):
        """Test financial metrics extraction"""
        text = """
        The company reported revenue of $2.5 billion for Q4 2024, 
        with a profit margin of 15.2% and market cap of $50 billion.
        The company employs 25,000 people globally.
        """
        
        metrics = analyze_financial_metrics(text)
        
        # Should find revenue, margin, market cap, employees
        assert len(metrics) >= 3
        
        # Check if we found revenue
        revenue_found = any(m["metric"].lower() == "revenue" for m in metrics)
        assert revenue_found
        
        # Check if we found market cap
        market_cap_found = any(m["metric"].lower() == "market_cap" for m in metrics)
        assert market_cap_found
    
    def test_analyze_financial_metrics_empty(self):
        """Test financial metrics with empty text"""
        metrics = analyze_financial_metrics("")
        assert metrics == []
    
    @patch('bi_core.tools._safe_get')
    def test_business_wiki_search_mock(self, mock_get):
        """Test Wikipedia search with mocked response"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "query": {
                "search": [
                    {
                        "title": "Apple Inc.",
                        "snippet": "Apple Inc. is an American multinational technology <span>company</span>",
                        "timestamp": "2024-01-01T00:00:00Z"
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        results = business_wiki_search("Apple company")
        
        assert len(results) >= 1
        assert results[0]["title"] == "Apple Inc."
        assert "company" in results[0]["snippet"]
        assert "wikipedia.org" in results[0]["url"]
    
    @patch('bi_core.tools._safe_get')
    def test_fetch_business_content_mock(self, mock_get):
        """Test content fetching with mocked response"""
        mock_response = MagicMock()
        mock_response.text = """
        <html>
            <body>
                <main>
                    <h1>Company News</h1>
                    <p>This is important business content about financial performance.</p>
                </main>
            </body>
        </html>
        """
        mock_get.return_value = mock_response
        
        content = fetch_business_content("https://example.com/news")
        
        assert "Company News" in content
        assert "business content" in content
        assert "financial performance" in content
    
    @patch('duckduckgo_search.DDGS')
    def test_business_web_search_mock(self, mock_ddgs):
        """Test web search with mocked DuckDuckGo"""
        mock_ddgs_instance = MagicMock()
        mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance
        
        mock_ddgs_instance.text.return_value = [
            {
                "title": "Tesla Q4 2024 Earnings Report",
                "body": "Tesla reported strong quarterly earnings with revenue growth of 25%",
                "href": "https://investor.tesla.com/earnings"
            }
        ]
        
        results = business_web_search("Tesla earnings Q4 2024")
        
        assert len(results) >= 1
        assert "Tesla" in results[0]["title"]
        assert "earnings" in results[0]["snippet"]
        assert results[0]["relevance_score"] > 0
    
    @patch('duckduckgo_search.DDGS')
    def test_company_news_search_mock(self, mock_ddgs):
        """Test company news search with mocked response"""
        mock_ddgs_instance = MagicMock()
        mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance
        
        mock_ddgs_instance.text.return_value = [
            {
                "title": "Microsoft announces new partnership",
                "body": "Microsoft announced a strategic partnership to expand AI capabilities",
                "href": "https://news.microsoft.com/partnership"
            }
        ]
        
        results = company_news_search("Microsoft")
        
        assert len(results) >= 1
        assert "Microsoft" in results[0]["title"]
        assert results[0]["relevance_score"] > 0
    
    def test_market_data_search_structure(self):
        """Test market data search return structure"""
        # This test doesn't mock external calls but checks the structure
        result = market_data_search("AAPL")
        
        assert "company" in result
        assert "search_results" in result
        assert "extracted_data" in result
        assert result["company"] == "AAPL"
    
class TestToolIntegration:
    """Test tool integration scenarios"""
    
    @patch('bi_core.tools._safe_get')
    @patch('duckduckgo_search.DDGS')
    def test_comprehensive_company_research(self, mock_ddgs, mock_get):
        """Test a comprehensive company research scenario"""
        # Mock Wikipedia response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "query": {
                "search": [
                    {
                        "title": "Amazon.com",
                        "snippet": "Amazon.com, Inc. is an American multinational technology company",
                        "timestamp": "2024-01-01T00:00:00Z"
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        # Mock DuckDuckGo response
        mock_ddgs_instance = MagicMock()
        mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance
        mock_ddgs_instance.text.return_value = [
            {
                "title": "Amazon Reports Strong Q4 Results",
                "body": "Amazon reported revenue of $170 billion with AWS growing 20%",
                "href": "https://amazon.com/investor-relations"
            }
        ]
        
        # Test Wikipedia search
        wiki_results = business_wiki_search("Amazon company")
        assert len(wiki_results) >= 1
        assert "Amazon" in wiki_results[0]["title"]
        
        # Test web search
        web_results = business_web_search("Amazon financial results 2024")
        assert len(web_results) >= 1
        
        # Test news search
        news_results = company_news_search("Amazon")
        assert len(news_results) >= 1
        
        # Test metrics extraction from news
        news_text = web_results[0]["snippet"]
        metrics = analyze_financial_metrics(news_text)
        # Should find some financial metrics
        assert isinstance(metrics, list)

if __name__ == "__main__":
    pytest.main([__file__])
