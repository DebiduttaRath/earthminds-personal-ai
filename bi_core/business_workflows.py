"""
Specialized Business Intelligence Workflows
Pre-defined workflows for common business intelligence tasks
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
from enum import Enum

from bi_core.graph import build_business_graph, create_business_thread
from bi_core.settings import settings
from bi_core.telemetry import get_logger
from langchain_core.messages import HumanMessage

logger = get_logger(__name__)

class AnalysisComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class BusinessIntelligenceWorkflow:
    """Main class for orchestrating business intelligence workflows"""
    
    def __init__(self):
        self.graph = build_business_graph()
        self.active_workflows = {}
        
    def analyze_query_complexity(self, query: str) -> AnalysisComplexity:
        """Analyze the complexity of a business query"""
        
        # Simple queries (single entity, basic info)
        simple_indicators = [
            "what is", "who is", "when was", "where is",
            "stock price", "headquarters", "founded"
        ]
        
        # Complex queries (multi-step analysis, comparisons)
        complex_indicators = [
            "compare", "analyze", "evaluate", "assess", "pros and cons",
            "investment recommendation", "competitive landscape", "market analysis",
            "financial health", "strategic position", "due diligence"
        ]
        
        query_lower = query.lower()
        
        # Check for complex indicators first
        if any(indicator in query_lower for indicator in complex_indicators):
            return AnalysisComplexity.COMPLEX
        
        # Check for simple indicators
        if any(indicator in query_lower for indicator in simple_indicators):
            return AnalysisComplexity.SIMPLE
        
        # Default to moderate for everything else
        return AnalysisComplexity.MODERATE
    
    async def execute_market_research_workflow(
        self,
        market_or_industry: str,
        focus_areas: Optional[List[str]] = None,
        geographic_scope: str = "Global",
        time_horizon: str = "2024-2025"
    ) -> Dict[str, Any]:
        """Execute comprehensive market research workflow"""
        
        workflow_id = f"market_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_workflows[workflow_id] = {"status": WorkflowStatus.RUNNING, "type": "Market Research"}
        
        try:
            logger.info(f"Starting market research workflow for: {market_or_industry}")
            
            focus_text = f" focusing on {', '.join(focus_areas)}" if focus_areas else ""
            query = f"""
            Conduct comprehensive market research for the {market_or_industry} industry{focus_text}.
            
            Geographic Scope: {geographic_scope}
            Time Horizon: {time_horizon}
            
            Please provide:
            1. Market size and growth projections
            2. Key market segments and dynamics
            3. Major players and competitive landscape
            4. Technology trends and disruptions
            5. Regulatory environment and impacts
            6. Investment opportunities and risks
            7. Future outlook and strategic recommendations
            
            Include specific data points, financial metrics, and cite all sources.
            """
            
            config = create_business_thread(workflow_id)
            
            # Execute the workflow
            result = await self._execute_workflow(query, config, "Market Research")
            
            self.active_workflows[workflow_id]["status"] = WorkflowStatus.COMPLETED
            logger.info(f"Market research workflow completed: {workflow_id}")
            
            return result
            
        except Exception as e:
            self.active_workflows[workflow_id]["status"] = WorkflowStatus.FAILED
            logger.error(f"Market research workflow failed: {e}")
            raise
    
    async def execute_competitive_analysis_workflow(
        self,
        target_company: str,
        competitors: Optional[List[str]] = None,
        analysis_dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute competitive analysis workflow"""
        
        workflow_id = f"competitive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_workflows[workflow_id] = {"status": WorkflowStatus.RUNNING, "type": "Competitive Analysis"}
        
        try:
            logger.info(f"Starting competitive analysis for: {target_company}")
            
            competitors_text = f" Compare against: {', '.join(competitors)}" if competitors else ""
            dimensions_text = f" Focus on: {', '.join(analysis_dimensions)}" if analysis_dimensions else ""
            
            query = f"""
            Perform comprehensive competitive analysis for {target_company}.{competitors_text}{dimensions_text}
            
            Please provide:
            1. Company profile and business model analysis
            2. Market positioning and competitive advantages
            3. Financial performance comparison
            4. Product/service portfolio analysis
            5. Strengths, weaknesses, opportunities, threats (SWOT)
            6. Strategic initiatives and partnerships
            7. Competitive response recommendations
            
            Include quantitative metrics, market share data, and financial comparisons where available.
            """
            
            config = create_business_thread(workflow_id)
            result = await self._execute_workflow(query, config, "Competitive Analysis")
            
            self.active_workflows[workflow_id]["status"] = WorkflowStatus.COMPLETED
            logger.info(f"Competitive analysis completed: {workflow_id}")
            
            return result
            
        except Exception as e:
            self.active_workflows[workflow_id]["status"] = WorkflowStatus.FAILED
            logger.error(f"Competitive analysis failed: {e}")
            raise
    
    async def execute_investment_screening_workflow(
        self,
        company_or_sector: str,
        investment_criteria: Optional[Dict[str, Any]] = None,
        risk_tolerance: str = "Moderate"
    ) -> Dict[str, Any]:
        """Execute investment screening and analysis workflow"""
        
        workflow_id = f"investment_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_workflows[workflow_id] = {"status": WorkflowStatus.RUNNING, "type": "Investment Screening"}
        
        try:
            logger.info(f"Starting investment screening for: {company_or_sector}")
            
            criteria_text = ""
            if investment_criteria:
                criteria_text = f"\nInvestment Criteria: {', '.join([f'{k}: {v}' for k, v in investment_criteria.items()])}"
            
            query = f"""
            Conduct thorough investment analysis and screening for {company_or_sector}.
            
            Risk Tolerance: {risk_tolerance}{criteria_text}
            
            Please provide:
            1. Investment thesis and key value drivers
            2. Business model and revenue sustainability analysis
            3. Financial health assessment (profitability, cash flow, debt)
            4. Market position and competitive advantages
            5. Growth prospects and scalability
            6. Risk assessment (business, market, financial, regulatory)
            7. Valuation analysis and peer comparison
            8. Investment recommendation with price targets
            9. Scenario analysis (bull, base, bear cases)
            
            Include specific financial ratios, multiples, and quantitative metrics.
            Use chain-of-thought reasoning for complex financial analysis.
            """
            
            config = create_business_thread(workflow_id)
            result = await self._execute_workflow(query, config, "Investment Screening")
            
            self.active_workflows[workflow_id]["status"] = WorkflowStatus.COMPLETED
            logger.info(f"Investment screening completed: {workflow_id}")
            
            return result
            
        except Exception as e:
            self.active_workflows[workflow_id]["status"] = WorkflowStatus.FAILED
            logger.error(f"Investment screening failed: {e}")
            raise
    
    async def execute_trend_analysis_workflow(
        self,
        trend_topic: str,
        time_horizon: str = "Next 2-3 years",
        industries_affected: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute trend analysis workflow"""
        
        workflow_id = f"trend_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_workflows[workflow_id] = {"status": WorkflowStatus.RUNNING, "type": "Trend Analysis"}
        
        try:
            logger.info(f"Starting trend analysis for: {trend_topic}")
            
            industries_text = f" Impact on industries: {', '.join(industries_affected)}" if industries_affected else ""
            
            query = f"""
            Analyze the business and market implications of the trend: {trend_topic}
            
            Time Horizon: {time_horizon}{industries_text}
            
            Please provide:
            1. Trend definition and current state
            2. Market size and adoption metrics
            3. Key drivers and enablers
            4. Industry disruption analysis
            5. Leading companies and innovations
            6. Investment landscape and funding trends
            7. Challenges and barriers to adoption
            8. Future scenarios and timeline predictions
            9. Strategic recommendations for businesses
            
            Include quantitative data on market size, growth rates, and investment flows.
            """
            
            config = create_business_thread(workflow_id)
            result = await self._execute_workflow(query, config, "Trend Analysis")
            
            self.active_workflows[workflow_id]["status"] = WorkflowStatus.COMPLETED
            logger.info(f"Trend analysis completed: {workflow_id}")
            
            return result
            
        except Exception as e:
            self.active_workflows[workflow_id]["status"] = WorkflowStatus.FAILED
            logger.error(f"Trend analysis failed: {e}")
            raise
    
    async def execute_company_intelligence_workflow(
        self,
        company_name: str,
        intelligence_depth: str = "Comprehensive",
        include_financials: bool = True,
        include_news: bool = True
    ) -> Dict[str, Any]:
        """Execute company intelligence gathering workflow"""
        
        workflow_id = f"company_intelligence_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_workflows[workflow_id] = {"status": WorkflowStatus.RUNNING, "type": "Company Intelligence"}
        
        try:
            logger.info(f"Starting company intelligence gathering for: {company_name}")
            
            financial_text = " Include detailed financial analysis." if include_financials else ""
            news_text = " Include recent news and developments." if include_news else ""
            
            query = f"""
            Gather comprehensive company intelligence for {company_name}.
            
            Depth: {intelligence_depth}{financial_text}{news_text}
            
            Please provide:
            1. Company overview and business model
            2. Leadership team and key personnel
            3. Corporate structure and ownership
            4. Products/services portfolio
            5. Financial performance and key metrics
            6. Market position and competitive landscape
            7. Recent developments and strategic initiatives
            8. Partnerships and key relationships
            9. Risk factors and challenges
            10. Future outlook and strategic direction
            
            Include specific data points, financial figures, and recent news citations.
            """
            
            config = create_business_thread(workflow_id)
            result = await self._execute_workflow(query, config, "Company Intelligence")
            
            self.active_workflows[workflow_id]["status"] = WorkflowStatus.COMPLETED
            logger.info(f"Company intelligence completed: {workflow_id}")
            
            return result
            
        except Exception as e:
            self.active_workflows[workflow_id]["status"] = WorkflowStatus.FAILED
            logger.error(f"Company intelligence failed: {e}")
            raise
    
    async def _execute_workflow(self, query: str, config: Dict[str, str], analysis_type: str) -> Dict[str, Any]:
        """Internal method to execute a workflow"""
        
        messages = [HumanMessage(content=query)]
        
        # Execute the graph workflow
        result = {"messages": [], "sources": [], "extracted_data": {}, "recommendations": []}
        
        async for event in self.graph.astream(
            {"messages": messages, "analysis_type": analysis_type},
            config,
            stream_mode="values"
        ):
            # Collect all the workflow results
            if "messages" in event:
                result["messages"].extend(event["messages"])
            if "sources" in event:
                result["sources"] = event["sources"]
            if "extracted_data" in event:
                result["extracted_data"] = event["extracted_data"]
            if "recommendations" in event:
                result["recommendations"] = event["recommendations"]
            if "confidence_score" in event:
                result["confidence_score"] = event["confidence_score"]
        
        return result
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific workflow"""
        return self.active_workflows.get(workflow_id)
    
    def list_active_workflows(self) -> Dict[str, Dict[str, Any]]:
        """List all active workflows"""
        return {k: v for k, v in self.active_workflows.items() 
                if v["status"] in [WorkflowStatus.RUNNING, WorkflowStatus.PENDING]}
    
    def cleanup_completed_workflows(self, hours_old: int = 24):
        """Clean up completed workflows older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours_old)
        
        to_remove = []
        for workflow_id, workflow_info in self.active_workflows.items():
            # Extract timestamp from workflow_id
            try:
                timestamp_str = workflow_id.split('_')[-1]
                workflow_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                
                if (workflow_time < cutoff_time and 
                    workflow_info["status"] in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]):
                    to_remove.append(workflow_id)
            except (ValueError, IndexError):
                # Skip workflows with invalid timestamp format
                continue
        
        for workflow_id in to_remove:
            del self.active_workflows[workflow_id]
        
        logger.info(f"Cleaned up {len(to_remove)} old workflows")

# Convenience functions for common workflows
async def quick_market_research(market: str) -> Dict[str, Any]:
    """Quick market research for a specific market"""
    workflow = BusinessIntelligenceWorkflow()
    return await workflow.execute_market_research_workflow(market)

async def quick_competitive_analysis(company: str) -> Dict[str, Any]:
    """Quick competitive analysis for a company"""
    workflow = BusinessIntelligenceWorkflow()
    return await workflow.execute_competitive_analysis_workflow(company)

async def quick_investment_screening(company: str) -> Dict[str, Any]:
    """Quick investment screening for a company"""
    workflow = BusinessIntelligenceWorkflow()
    return await workflow.execute_investment_screening_workflow(company)

# Export main classes and functions
__all__ = [
    "BusinessIntelligenceWorkflow",
    "AnalysisComplexity", 
    "WorkflowStatus",
    "quick_market_research",
    "quick_competitive_analysis", 
    "quick_investment_screening"
]
