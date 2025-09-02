"""
Business Intelligence Graph
LangGraph implementation for orchestrating business intelligence workflows
"""

from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json
from datetime import datetime

from bi_core.llm_factory import get_smart_llm, get_llm
from bi_core.prompts import (
    BUSINESS_SYSTEM_PROMPT, MARKET_RESEARCH_PROMPT, COMPETITIVE_ANALYSIS_PROMPT,
    INVESTMENT_SCREENING_PROMPT, COMPANY_INTELLIGENCE_PROMPT, TREND_ANALYSIS_PROMPT,
    FINANCIAL_ANALYSIS_PROMPT, SEARCH_DECISION_PROMPT
)
from bi_core.tools import BUSINESS_TOOLS
from bi_core.settings import settings
from bi_core.telemetry import get_logger

logger = get_logger(__name__)

class BusinessState(TypedDict):
    """State management for business intelligence workflows"""
    messages: Annotated[list, add_messages]
    analysis_type: str
    sources: List[Dict[str, str]]
    extracted_data: Dict[str, Any]
    reasoning_trace: str
    confidence_score: float
    recommendations: List[str]
    next_steps: List[str]

def determine_analysis_type(query: str) -> str:
    """Determine the type of business analysis needed based on the query"""
    query_lower = query.lower()
    
    type_keywords = {
        "Market Research": ["market research", "market size", "industry analysis", "market trends", "market overview"],
        "Competitive Analysis": ["competitive analysis", "competitor", "competition", "market share", "competitive landscape"],
        "Investment Screening": ["investment", "invest in", "stock analysis", "buy recommendation", "valuation"],
        "Company Intelligence": ["company profile", "about company", "company information", "business model"],
        "Trend Analysis": ["trends", "emerging", "future of", "predictions", "forecast"],
        "Financial Analysis": ["financial analysis", "financial performance", "revenue", "profit", "cash flow"]
    }
    
    for analysis_type, keywords in type_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return analysis_type
    
    return "Custom Query"

def should_use_reasoning_llm(query: str, analysis_type: str) -> bool:
    """Determine if we should use reasoning-capable LLM"""
    reasoning_triggers = [
        "analyze", "compare", "evaluate", "assess", "explain why", "pros and cons",
        "advantages", "disadvantages", "step by step", "reasoning", "complex"
    ]
    
    complex_analysis_types = ["Investment Screening", "Competitive Analysis", "Financial Analysis"]
    
    return (any(trigger in query.lower() for trigger in reasoning_triggers) or 
            analysis_type in complex_analysis_types or
            len(query.split()) > 20)

def business_analyzer(state: BusinessState) -> Dict[str, Any]:
    """Main business analysis node"""
    try:
        messages = state["messages"]
        analysis_type = state.get("analysis_type", "Custom Query")
        
        # Get the latest user message
        user_message = ""
        for msg in reversed(messages):
            if msg.type == "human":
                user_message = str(msg.content)
                break
        
        logger.info(f"Analyzing business query: {analysis_type}")
        
        # Determine if analysis type wasn't set
        if analysis_type == "Custom Query":
            analysis_type = determine_analysis_type(user_message)
        
        # Select appropriate LLM based on complexity
        use_reasoning = should_use_reasoning_llm(user_message, analysis_type)
        llm = get_smart_llm(user_message) if use_reasoning else get_llm()
        
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(BUSINESS_TOOLS)
        
        # Build context with appropriate prompts
        system_prompt = BUSINESS_SYSTEM_PROMPT
        
        # Add specific analysis prompt based on type
        analysis_prompts = {
            "Market Research": MARKET_RESEARCH_PROMPT,
            "Competitive Analysis": COMPETITIVE_ANALYSIS_PROMPT,
            "Investment Screening": INVESTMENT_SCREENING_PROMPT,
            "Company Intelligence": COMPANY_INTELLIGENCE_PROMPT,
            "Trend Analysis": TREND_ANALYSIS_PROMPT,
            "Financial Analysis": FINANCIAL_ANALYSIS_PROMPT
        }
        
        if analysis_type in analysis_prompts:
            system_prompt += "\n\n" + analysis_prompts[analysis_type]
        
        # Add search decision guidance
        system_prompt += "\n\n" + SEARCH_DECISION_PROMPT
        
        # Prepare messages with system prompt
        if not any(msg.type == "system" for msg in messages):
            messages = [SystemMessage(content=system_prompt)] + messages
        
        # Add analysis type context
        enhanced_query = f"[Analysis Type: {analysis_type}]\n\n{user_message}"
        messages.append(HumanMessage(content=enhanced_query))
        
        # Generate response
        response = llm_with_tools.invoke(messages)
        
        # Extract reasoning traces if present
        reasoning_trace = ""
        if hasattr(response, 'content') and isinstance(response.content, str):
            content = response.content
            if "<think>" in content and "</think>" in content:
                start = content.find("<think>") + 7
                end = content.find("</think>")
                reasoning_trace = content[start:end].strip()
        
        logger.info(f"Business analysis completed for {analysis_type}")
        
        return {
            "messages": [response],
            "analysis_type": analysis_type,
            "reasoning_trace": reasoning_trace,
            "confidence_score": 0.8  # Default confidence, could be enhanced with actual scoring
        }
        
    except Exception as e:
        logger.error(f"Business analysis failed: {e}")
        error_msg = AIMessage(content=f"I apologize, but I encountered an error during analysis: {str(e)}. Please try rephrasing your question or contact support.")
        return {"messages": [error_msg]}

def research_coordinator(state: BusinessState) -> Dict[str, Any]:
    """Coordinate research activities and tool usage"""
    try:
        messages = state["messages"]
        
        # Check if we have recent tool calls
        recent_tool_calls = []
        for msg in reversed(messages[-5:]):  # Check last 5 messages
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                recent_tool_calls.extend(msg.tool_calls)
        
        if not recent_tool_calls:
            # No tools were called, continue to end
            return {"messages": []}
        
        logger.info(f"Research coordinator processing {len(recent_tool_calls)} tool calls")
        
        # Extract and organize tool results
        sources = []
        extracted_data = {}
        
        # Look for tool responses in recent messages
        for msg in reversed(messages[-10:]):
            if msg.type == "tool":
                tool_name = getattr(msg, 'name', 'unknown_tool')
                tool_content = str(msg.content)
                
                # Parse different tool responses
                if tool_name in ['business_wiki_search', 'business_web_search', 'company_news_search']:
                    try:
                        # Try to parse as JSON (list of search results)
                        if tool_content.startswith('['):
                            results = json.loads(tool_content)
                            for result in results:
                                if isinstance(result, dict) and result.get('url'):
                                    sources.append({
                                        'title': result.get('title', 'Unknown'),
                                        'url': result.get('url', ''),
                                        'snippet': result.get('snippet', ''),
                                        'tool_used': tool_name
                                    })
                    except json.JSONDecodeError:
                        # Handle as plain text
                        sources.append({
                            'title': f"Result from {tool_name}",
                            'url': '',
                            'snippet': tool_content[:200] + "..." if len(tool_content) > 200 else tool_content,
                            'tool_used': tool_name
                        })
                
                elif tool_name == 'analyze_financial_metrics':
                    try:
                        if tool_content.startswith('['):
                            metrics = json.loads(tool_content)
                            for metric in metrics:
                                if isinstance(metric, dict):
                                    extracted_data[metric.get('metric', 'Unknown')] = {
                                        'value': metric.get('value', ''),
                                        'period': metric.get('period', ''),
                                        'currency': metric.get('currency', '')
                                    }
                    except json.JSONDecodeError:
                        extracted_data['financial_summary'] = tool_content[:500]
        
        # Remove duplicates from sources
        unique_sources = []
        seen_urls = set()
        for source in sources:
            if source['url'] not in seen_urls:
                seen_urls.add(source['url'])
                unique_sources.append(source)
        
        logger.info(f"Research coordination completed: {len(unique_sources)} sources, {len(extracted_data)} data points")
        
        return {
            "sources": unique_sources[:settings.max_sources_per_analysis],
            "extracted_data": extracted_data
        }
        
    except Exception as e:
        logger.error(f"Research coordination failed: {e}")
        return {"sources": [], "extracted_data": {}}

def synthesis_node(state: BusinessState) -> Dict[str, Any]:
    """Synthesize research results into final recommendations"""
    try:
        messages = state["messages"]
        analysis_type = state.get("analysis_type", "Custom Query")
        sources = state.get("sources", [])
        extracted_data = state.get("extracted_data", {})
        
        logger.info(f"Synthesizing results for {analysis_type}")
        
        # Create synthesis prompt
        synthesis_context = f"""
Based on the research conducted, please provide a comprehensive synthesis for this {analysis_type} analysis.

Available Data:
- {len(sources)} research sources
- {len(extracted_data)} extracted data points

Research Sources:
{chr(10).join([f"- {source['title']}: {source['snippet'][:100]}..." for source in sources[:5]])}

Extracted Data:
{json.dumps(extracted_data, indent=2) if extracted_data else "No structured data extracted"}

Please provide:
1. Executive Summary (3-5 key insights)
2. Strategic Recommendations (3-5 actionable items)
3. Next Steps for further analysis
4. Confidence assessment of the analysis

Format your response professionally and cite sources where relevant.
"""
        
        # Get LLM for synthesis (use reasoning for complex analysis types)
        use_reasoning = analysis_type in ["Investment Screening", "Competitive Analysis", "Financial Analysis"]
        llm = get_llm(for_reasoning=use_reasoning)
        
        synthesis_message = HumanMessage(content=synthesis_context)
        response = llm.invoke([synthesis_message])
        
        # Extract recommendations and next steps
        recommendations = []
        next_steps = []
        
        if hasattr(response, 'content'):
            content = str(response.content)
            
            # Simple extraction of recommendations and next steps
            if "recommendations" in content.lower():
                rec_section = content.lower().split("recommendations")[1].split("next steps")[0] if "next steps" in content.lower() else content.lower().split("recommendations")[1]
                recommendations = [line.strip() for line in rec_section.split('\n') if line.strip() and (line.strip().startswith('-') or line.strip().startswith('•'))]
            
            if "next steps" in content.lower():
                next_section = content.lower().split("next steps")[1]
                next_steps = [line.strip() for line in next_section.split('\n') if line.strip() and (line.strip().startswith('-') or line.strip().startswith('•'))]
        
        logger.info(f"Synthesis completed: {len(recommendations)} recommendations, {len(next_steps)} next steps")
        
        return {
            "messages": [response],
            "recommendations": recommendations[:5],
            "next_steps": next_steps[:5],
            "confidence_score": min(0.9, 0.5 + (len(sources) * 0.1) + (len(extracted_data) * 0.05))
        }
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        error_msg = AIMessage(content=f"Synthesis error: {str(e)}")
        return {"messages": [error_msg]}

def build_business_graph(checkpointer=None):
    """Build the business intelligence workflow graph"""
    
    # Initialize checkpointer if not provided
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    # Create the graph
    workflow = StateGraph(BusinessState)
    
    # Add nodes
    workflow.add_node("analyzer", business_analyzer)
    workflow.add_node("tools", ToolNode(BUSINESS_TOOLS))
    workflow.add_node("coordinator", research_coordinator)
    workflow.add_node("synthesis", synthesis_node)
    
    # Define the flow
    workflow.add_edge(START, "analyzer")
    
    # Conditional edge from analyzer
    workflow.add_conditional_edges(
        "analyzer",
        tools_condition,
        {
            "tools": "tools",
            END: "synthesis"
        }
    )
    
    # From tools back to coordinator
    workflow.add_edge("tools", "coordinator")
    
    # From coordinator to synthesis
    workflow.add_edge("coordinator", "synthesis")
    
    # From synthesis to end
    workflow.add_edge("synthesis", END)
    
    # Compile the graph
    graph = workflow.compile(checkpointer=checkpointer)
    
    logger.info("Business intelligence graph compiled successfully")
    return graph

def create_business_thread(thread_id: str = None) -> Dict[str, str]:
    """Create a new business intelligence thread"""
    if not thread_id:
        thread_id = f"bi-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    return {"configurable": {"thread_id": thread_id}}

# Export main functions
__all__ = [
    "build_business_graph",
    "create_business_thread", 
    "BusinessState",
    "determine_analysis_type"
]
