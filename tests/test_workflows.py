"""
Test suite for Business Intelligence Workflows
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from bi_core.business_workflows import (
    BusinessIntelligenceWorkflow, AnalysisComplexity, WorkflowStatus,
    quick_market_research, quick_competitive_analysis, quick_investment_screening
)

class TestWorkflowAnalysis:
    """Test workflow analysis and complexity detection"""
    
    def test_query_complexity_simple(self):
        """Test simple query complexity detection"""
        workflow = BusinessIntelligenceWorkflow()
        
        simple_queries = [
            "What is Apple Inc?",
            "Who is the CEO of Microsoft?",
            "When was Google founded?",
            "What is Tesla's stock price?"
        ]
        
        for query in simple_queries:
            complexity = workflow.analyze_query_complexity(query)
            assert complexity == AnalysisComplexity.SIMPLE, f"Query should be simple: {query}"
    
    def test_query_complexity_complex(self):
        """Test complex query complexity detection"""
        workflow = BusinessIntelligenceWorkflow()
        
        complex_queries = [
            "Compare Amazon and Microsoft's cloud computing strategies",
            "Analyze Tesla's competitive position in the EV market",
            "Evaluate Apple as an investment opportunity",
            "Assess the pros and cons of investing in renewable energy stocks"
        ]
        
        for query in complex_queries:
            complexity = workflow.analyze_query_complexity(query)
            assert complexity == AnalysisComplexity.COMPLEX, f"Query should be complex: {query}"
    
    def test_query_complexity_moderate(self):
        """Test moderate query complexity detection"""
        workflow = BusinessIntelligenceWorkflow()
        
        moderate_queries = [
            "Tell me about the smartphone industry",
            "What are the latest developments in AI?",
            "How is Netflix performing this year?"
        ]
        
        for query in moderate_queries:
            complexity = workflow.analyze_query_complexity(query)
            assert complexity == AnalysisComplexity.MODERATE, f"Query should be moderate: {query}"

class TestWorkflowExecution:
    """Test workflow execution scenarios"""
    
    @patch('bi_core.business_workflows.build_business_graph')
    @pytest.mark.asyncio
    async def test_market_research_workflow_structure(self, mock_build_graph):
        """Test market research workflow structure"""
        # Mock the graph
        mock_graph = MagicMock()
        mock_graph.astream = AsyncMock(return_value=iter([
            {
                "messages": [MagicMock(content="Market research analysis complete")],
                "sources": [{"title": "Market Report", "url": "https://example.com"}],
                "extracted_data": {"market_size": "$100B"},
                "recommendations": ["Recommendation 1"],
                "confidence_score": 0.85
            }
        ]))
        mock_build_graph.return_value = mock_graph
        
        workflow = BusinessIntelligenceWorkflow()
        
        result = await workflow.execute_market_research_workflow(
            market_or_industry="Electric Vehicles",
            focus_areas=["Market Size", "Key Players"],
            geographic_scope="North America"
        )
        
        assert "messages" in result
        assert "sources" in result
        assert "extracted_data" in result
        assert "recommendations" in result
        assert "confidence_score" in result
        
        # Check that the workflow was recorded
        assert len(workflow.active_workflows) > 0
        workflow_info = list(workflow.active_workflows.values())[0]
        assert workflow_info["type"] == "Market Research"
        assert workflow_info["status"] == WorkflowStatus.COMPLETED
    
    @patch('bi_core.business_workflows.build_business_graph')
    @pytest.mark.asyncio
    async def test_competitive_analysis_workflow_structure(self, mock_build_graph):
        """Test competitive analysis workflow structure"""
        # Mock the graph
        mock_graph = MagicMock()
        mock_graph.astream = AsyncMock(return_value=iter([
            {
                "messages": [MagicMock(content="Competitive analysis complete")],
                "sources": [{"title": "Company Report", "url": "https://example.com"}],
                "extracted_data": {"market_share": "25%"},
                "recommendations": ["Strategic recommendation"],
                "confidence_score": 0.78
            }
        ]))
        mock_build_graph.return_value = mock_graph
        
        workflow = BusinessIntelligenceWorkflow()
        
        result = await workflow.execute_competitive_analysis_workflow(
            target_company="Tesla",
            competitors=["BMW", "Mercedes", "Audi"],
            analysis_dimensions=["Technology", "Market Share", "Pricing"]
        )
        
        assert "messages" in result
        assert "sources" in result
        assert "extracted_data" in result
        assert "recommendations" in result
        
        # Check workflow recording
        assert len(workflow.active_workflows) > 0
        workflow_info = list(workflow.active_workflows.values())[0]
        assert workflow_info["type"] == "Competitive Analysis"
    
    @patch('bi_core.business_workflows.build_business_graph')
    @pytest.mark.asyncio
    async def test_investment_screening_workflow_structure(self, mock_build_graph):
        """Test investment screening workflow structure"""
        # Mock the graph
        mock_graph = MagicMock()
        mock_graph.astream = AsyncMock(return_value=iter([
            {
                "messages": [MagicMock(content="Investment analysis complete")],
                "sources": [{"title": "Financial Report", "url": "https://example.com"}],
                "extracted_data": {"pe_ratio": "15.5", "debt_ratio": "0.3"},
                "recommendations": ["Buy recommendation"],
                "confidence_score": 0.82
            }
        ]))
        mock_build_graph.return_value = mock_graph
        
        workflow = BusinessIntelligenceWorkflow()
        
        result = await workflow.execute_investment_screening_workflow(
            company_or_sector="Apple Inc",
            investment_criteria={"min_revenue": "100B", "max_debt_ratio": "0.4"},
            risk_tolerance="Conservative"
        )
        
        assert "messages" in result
        assert "sources" in result
        assert "extracted_data" in result
        assert "recommendations" in result
        
        # Check workflow recording
        workflow_info = list(workflow.active_workflows.values())[0]
        assert workflow_info["type"] == "Investment Screening"
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self):
        """Test workflow error handling"""
        workflow = BusinessIntelligenceWorkflow()
        
        # Mock graph to raise an exception
        with patch.object(workflow, 'graph') as mock_graph:
            mock_graph.astream.side_effect = Exception("Test error")
            
            with pytest.raises(Exception):
                await workflow.execute_market_research_workflow("Test Market")
            
            # Check that workflow status was updated to failed
            failed_workflows = [w for w in workflow.active_workflows.values() 
                             if w["status"] == WorkflowStatus.FAILED]
            assert len(failed_workflows) > 0

class TestWorkflowManagement:
    """Test workflow management features"""
    
    def test_workflow_status_tracking(self):
        """Test workflow status tracking"""
        workflow = BusinessIntelligenceWorkflow()
        
        # Add some test workflows
        workflow.active_workflows["test_1"] = {
            "status": WorkflowStatus.RUNNING,
            "type": "Market Research"
        }
        workflow.active_workflows["test_2"] = {
            "status": WorkflowStatus.COMPLETED,
            "type": "Competitive Analysis"
        }
        workflow.active_workflows["test_3"] = {
            "status": WorkflowStatus.FAILED,
            "type": "Investment Screening"
        }
        
        # Test status retrieval
        status = workflow.get_workflow_status("test_1")
        assert status["status"] == WorkflowStatus.RUNNING
        
        # Test active workflows listing
        active = workflow.list_active_workflows()
        assert len(active) == 1  # Only running workflows
        assert "test_1" in active
    
    def test_workflow_cleanup(self):
        """Test workflow cleanup functionality"""
        workflow = BusinessIntelligenceWorkflow()
        
        # Add workflows with timestamps (simulating old workflows)
        workflow.active_workflows["market_research_20240101_120000"] = {
            "status": WorkflowStatus.COMPLETED,
            "type": "Market Research"
        }
        workflow.active_workflows["competitive_analysis_20240102_150000"] = {
            "status": WorkflowStatus.FAILED,
            "type": "Competitive Analysis"
        }
        workflow.active_workflows["trend_analysis_20250201_100000"] = {
            "status": WorkflowStatus.RUNNING,
            "type": "Trend Analysis"
        }
        
        initial_count = len(workflow.active_workflows)
        
        # Cleanup old workflows (more than 24 hours old)
        workflow.cleanup_completed_workflows(hours_old=1)
        
        # Should have removed old completed/failed workflows but kept running ones
        assert len(workflow.active_workflows) < initial_count
        
        # Running workflow should still be there
        running_workflows = [w for w in workflow.active_workflows.values() 
                           if w["status"] == WorkflowStatus.RUNNING]
        assert len(running_workflows) >= 1

class TestQuickWorkflows:
    """Test quick workflow convenience functions"""
    
    @patch('bi_core.business_workflows.BusinessIntelligenceWorkflow.execute_market_research_workflow')
    @pytest.mark.asyncio
    async def test_quick_market_research(self, mock_execute):
        """Test quick market research function"""
        mock_execute.return_value = {"status": "completed"}
        
        result = await quick_market_research("AI Technology")
        
        mock_execute.assert_called_once_with("AI Technology")
        assert result["status"] == "completed"
    
    @patch('bi_core.business_workflows.BusinessIntelligenceWorkflow.execute_competitive_analysis_workflow')
    @pytest.mark.asyncio
    async def test_quick_competitive_analysis(self, mock_execute):
        """Test quick competitive analysis function"""
        mock_execute.return_value = {"status": "completed"}
        
        result = await quick_competitive_analysis("Tesla")
        
        mock_execute.assert_called_once_with("Tesla")
        assert result["status"] == "completed"
    
    @patch('bi_core.business_workflows.BusinessIntelligenceWorkflow.execute_investment_screening_workflow')
    @pytest.mark.asyncio
    async def test_quick_investment_screening(self, mock_execute):
        """Test quick investment screening function"""
        mock_execute.return_value = {"status": "completed"}
        
        result = await quick_investment_screening("AAPL")
        
        mock_execute.assert_called_once_with("AAPL")
        assert result["status"] == "completed"

if __name__ == "__main__":
    pytest.main([__file__])
