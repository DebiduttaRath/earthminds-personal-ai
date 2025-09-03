"""
🚀 Business Intelligence Platform - Enhanced & Redesigned
Multi-LLM system with advanced analytics, real-time data, and interactive UI
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
import logging
import traceback
from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('business_intelligence.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import BI components
try:
    from bi_core.graph import build_business_graph
    from bi_core.settings import settings
    from bi_core.business_workflows import BusinessIntelligenceWorkflow
    from bi_core.telemetry import setup_telemetry, get_logger
    from bi_core.memory_optimizer import memory_optimizer
    from bi_core.anti_hallucination import verify_analysis_reliability
    from bi_core.llm_factory import llm_factory
    logger.info("✅ All BI core components imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import BI components: {e}")
    st.error(f"Configuration Error: {e}")

# Initialize telemetry
try:
    setup_telemetry()
    bi_logger = get_logger(__name__)
    logger.info("✅ Telemetry system initialized")
except Exception as e:
    logger.warning(f"⚠️ Telemetry initialization failed: {e}")
    bi_logger = logger

# Page configuration with modern styling
st.set_page_config(
    page_title="🚀 Business Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/business-intelligence',
        'Report a bug': 'https://github.com/your-repo/business-intelligence/issues',
        'About': "Advanced Business Intelligence Platform powered by multiple LLMs"
    }
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .analysis-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .status-healthy { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-error { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Initialize session state with better structure
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "graph": None,
        "cfg": {"configurable": {"thread_id": settings.thread_id}},
        "workflow": None,
        "analysis_history": [],
        "reliability_reports": [],
        "memory_stats": [],
        "current_analysis": None,
        "system_health": {"status": "checking", "last_check": None}
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# Initialize core components
@st.cache_resource
def initialize_business_graph():
    """Initialize and cache the business graph"""
    try:
        graph = build_business_graph()
        logger.info("✅ Business intelligence graph initialized")
        return graph
    except Exception as e:
        logger.error(f"❌ Failed to initialize business graph: {e}")
        return None

@st.cache_resource
def initialize_workflow():
    """Initialize and cache the workflow"""
    try:
        workflow = BusinessIntelligenceWorkflow()
        logger.info("✅ Business workflow initialized")
        return workflow
    except Exception as e:
        logger.error(f"❌ Failed to initialize workflow: {e}")
        return None

# Load cached components
if st.session_state.graph is None:
    st.session_state.graph = initialize_business_graph()

if st.session_state.workflow is None:
    st.session_state.workflow = initialize_workflow()

# System Health Check
def check_system_health():
    """Check the health of all system components"""
    health_status = {
        "timestamp": datetime.now(),
        "components": {}
    }
    
    # Check LLM backends
    try:
        for backend in ["groq", "ollama"]:  # Skip deepseek for now as it needs API key
            try:
                health_check = llm_factory.health_check(backend)
                health_status["components"][f"llm_{backend}"] = health_check
            except Exception as e:
                health_status["components"][f"llm_{backend}"] = {
                    "status": "unhealthy", 
                    "error": str(e)
                }
    except Exception as e:
        health_status["components"]["llm_system"] = {
            "status": "error", 
            "error": str(e)
        }
    
    # Check web scraping capabilities
    try:
        import requests
        response = requests.get("https://httpbin.org/status/200", timeout=5)
        health_status["components"]["web_scraping"] = {
            "status": "healthy" if response.status_code == 200 else "unhealthy"
        }
    except Exception as e:
        health_status["components"]["web_scraping"] = {
            "status": "unhealthy", 
            "error": str(e)
        }
    
    # Check memory optimizer
    try:
        mem_stats = memory_optimizer.get_memory_stats()
        health_status["components"]["memory_optimizer"] = {
            "status": "healthy",
            "memory_mb": mem_stats.get("rss_mb", 0),
            "cache_size": mem_stats.get("ttl_cache_size", 0)
        }
    except Exception as e:
        health_status["components"]["memory_optimizer"] = {
            "status": "unhealthy", 
            "error": str(e)
        }
    
    return health_status

# Header with modern design
st.markdown("""
<div class="main-header">
    <h1>🚀 Business Intelligence Platform</h1>
    <p>Advanced AI-Powered Business Analysis | Multi-LLM Architecture | Real-Time Data</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with enhanced configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # System Health Status
    with st.expander("🏥 System Health", expanded=True):
        if st.button("🔄 Check System Health"):
            with st.spinner("Checking system health..."):
                health = check_system_health()
                st.session_state.system_health = health
        
        if st.session_state.system_health["status"] != "checking":
            health = st.session_state.system_health
            
            for component, status in health["components"].items():
                if status.get("status") == "healthy":
                    st.markdown(f"✅ **{component.replace('_', ' ').title()}**: Healthy")
                elif status.get("status") == "unhealthy":
                    st.markdown(f"⚠️ **{component.replace('_', ' ').title()}**: Issues detected")
                else:
                    st.markdown(f"❌ **{component.replace('_', ' ').title()}**: Error")
    
    # LLM Backend Selection
    st.subheader("🤖 AI Backend")
    backend_options = ["groq", "ollama"]  # Start with working backends
    selected_backend = st.selectbox(
        "Primary Backend",
        backend_options,
        index=0,
        help="Groq: Fast inference | Ollama: Local processing"
    )
    
    # Model selection based on backend
    if selected_backend == "groq":
        model_options = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
        selected_model = st.selectbox("Model", model_options)
        st.info("🔑 Add GROQ_API_KEY for full functionality")
    else:
        model_options = ["llama3.2", "qwen2.5"]
        selected_model = st.selectbox("Local Model", model_options)
        st.info("🏠 Using local Ollama server")
    
    # Analysis type
    st.subheader("📊 Analysis Type")
    analysis_types = [
        "🏢 Market Research",
        "⚔️ Competitive Analysis", 
        "💰 Investment Screening",
        "🔍 Company Intelligence",
        "📈 Trend Analysis",
        "💹 Financial Analysis",
        "❓ Custom Query"
    ]
    analysis_type = st.selectbox("Analysis Focus", analysis_types)
    clean_analysis_type = analysis_type.split(" ", 1)[1]  # Remove emoji
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            reasoning_enabled = st.checkbox("🧠 Reasoning Traces", value=True)
            reliability_check = st.checkbox("🔍 Reliability Check", value=True)
        with col2:
            memory_optimization = st.checkbox("💾 Memory Optimization", value=True)
            enhanced_search = st.checkbox("🔍 Enhanced Search", value=True)
        
        max_tokens = st.slider("Max Tokens", 512, 4096, 2048)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.6, 0.1)
    
    # Session Statistics
    st.markdown("---")
    st.subheader("📈 Session Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Analyses", len(st.session_state.analysis_history))
        st.metric("Backend", selected_backend.upper())
    
    with col2:
        if memory_optimization and st.session_state.system_health.get("components", {}).get("memory_optimizer"):
            mem_info = st.session_state.system_health["components"]["memory_optimizer"]
            st.metric("Memory", f"{mem_info.get('memory_mb', 0):.1f} MB")
            st.metric("Cache", f"{mem_info.get('cache_size', 0)}")
    
    # Action buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.rerun()
    with col2:
        if st.button("💾 Export Data"):
            export_data = {
                "history": st.session_state.analysis_history,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "backend": selected_backend,
                    "model": selected_model,
                    "analysis_type": clean_analysis_type
                }
            }
            st.download_button(
                "📥 Download",
                json.dumps(export_data, indent=2),
                f"bi_session_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                "application/json"
            )

# Main Analysis Interface
st.header("🔍 Business Analysis Center")

# Create enhanced input interface
col1, col2 = st.columns([3, 1])

with col1:
    if clean_analysis_type == "Custom Query":
        query = st.text_area(
            "Enter your business question:",
            placeholder="e.g., Analyze Tesla's competitive position in the EV market and predict future growth opportunities",
            height=120,
            help="Ask any business question - our AI will determine the best analysis approach"
        )
    else:
        # Smart templates based on analysis type
        templates = {
            "Market Research": "Research the market size, growth trends, and key players in [INDUSTRY/MARKET]. Include market drivers, challenges, and 3-year outlook.",
            "Competitive Analysis": "Perform a comprehensive competitive analysis of [COMPANY] including direct competitors, market positioning, SWOT analysis, and strategic recommendations.",
            "Investment Screening": "Evaluate [COMPANY/SECTOR] as an investment opportunity. Analyze financial metrics, growth potential, risks, and provide investment thesis.",
            "Company Intelligence": "Gather comprehensive intelligence on [COMPANY] including business model, recent developments, financial performance, and strategic initiatives.",
            "Trend Analysis": "Analyze emerging trends in [INDUSTRY/TECHNOLOGY] including adoption rates, market impact, key drivers, and future predictions.",
            "Financial Analysis": "Conduct detailed financial analysis of [COMPANY] including revenue trends, profitability, debt analysis, and financial health assessment."
        }
        
        query = st.text_area(
            f"{clean_analysis_type} Query:",
            value=templates.get(clean_analysis_type, ""),
            height=120,
            help="Customize the template or use as-is. Replace [PLACEHOLDERS] with specific companies/industries."
        )

with col2:
    st.markdown("### 🚀 Quick Actions")
    
    # Enhanced analysis button
    analysis_ready = bool(query.strip())
    
    if st.button("🔍 **Run Analysis**", type="primary", disabled=not analysis_ready, use_container_width=True):
        if st.session_state.graph is None:
            st.error("❌ System not properly initialized. Please refresh the page.")
        else:
            # Start analysis
            st.session_state.current_analysis = {
                "query": query,
                "type": clean_analysis_type,
                "backend": selected_backend,
                "model": selected_model,
                "timestamp": datetime.now(),
                "status": "running"
            }
            st.rerun()
    
    # Additional quick actions
    if st.button("💡 **Get Suggestions**", use_container_width=True):
        suggestions = [
            "Analyze Apple's position in the smartphone market",
            "Research renewable energy investment opportunities",
            "Compare Netflix vs Disney+ streaming strategies",
            "Evaluate AI startup investment trends in 2024",
            "Analyze Tesla's supply chain challenges"
        ]
        st.info("💡 **Quick Ideas:**\n" + "\n".join([f"• {s}" for s in suggestions]))
    
    if st.button("📊 **Market Overview**", use_container_width=True):
        st.info("📊 **Today's Focus:**\n• Tech earnings season\n• Energy sector trends\n• Inflation impact analysis\n• ESG investment shifts")

# Analysis Execution and Results
if st.session_state.current_analysis and st.session_state.current_analysis["status"] == "running":
    analysis = st.session_state.current_analysis
    
    # Analysis container with better styling
    st.markdown(f"""
    <div class="analysis-container">
        <h3>🔬 {analysis['type']} Analysis</h3>
        <p><strong>Query:</strong> {analysis['query'][:100]}...</p>
        <p><strong>Backend:</strong> {analysis['backend'].upper()} | <strong>Model:</strong> {analysis['model']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress tracking
    progress_col1, progress_col2 = st.columns([3, 1])
    
    with progress_col1:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    with progress_col2:
        if st.button("⏹️ Stop Analysis"):
            st.session_state.current_analysis["status"] = "stopped"
            st.rerun()
    
    # Results containers
    results_container = st.container()
    reasoning_container = st.container()
    sources_container = st.container()
    
    try:
        # Execute the analysis
        status_text.text("🚀 Initializing analysis engine...")
        progress_bar.progress(0.1)
        
        messages = [HumanMessage(content=analysis["query"])]
        
        status_text.text("🔍 Processing query and gathering data...")
        progress_bar.progress(0.3)
        
        # Stream the analysis
        events = st.session_state.graph.stream(
            {"messages": messages, "analysis_type": analysis["type"]},
            st.session_state.cfg,
            stream_mode="values"
        )
        
        response_content = ""
        reasoning_content = ""
        sources = []
        metrics = []
        tool_logs = []
        
        event_count = 0
        for i, event in enumerate(events):
            event_count += 1
            progress_bar.progress(min(0.3 + (event_count * 0.1), 0.9))
            
            if "messages" in event and event["messages"]:
                msg = event["messages"][-1]
                
                if hasattr(msg, "content"):
                    content = str(msg.content)
                    
                    # Handle reasoning traces
                    if "<think>" in content and "</think>" in content:
                        think_start = content.find("<think>") + 7
                        think_end = content.find("</think>")
                        reasoning_content = content[think_start:think_end]
                        response_content = content.replace(f"<think>{reasoning_content}</think>", "").strip()
                    else:
                        response_content = content
            
            # Extract sources and tool calls
            if "sources" in event:
                sources = event["sources"]
            
            # Log tool executions
            if "tool_calls" in str(event) or "function_call" in str(event):
                tool_logs.append({
                    "timestamp": datetime.now().isoformat(),
                    "event": str(event)[:200] + "..." if len(str(event)) > 200 else str(event)
                })
                status_text.text("🛠️ Using analysis tools to gather data...")
            
            # Update status based on progress
            if response_content:
                status_text.text("🧠 Generating comprehensive analysis...")
        
        progress_bar.progress(0.9)
        status_text.text("🔍 Verifying information reliability...")
        
        # Reliability check
        reliability_report = None
        if reliability_check and response_content:
            try:
                reliability_report = verify_analysis_reliability(
                    response_content, sources, analysis["type"]
                )
                st.session_state.reliability_reports.append(reliability_report)
            except Exception as e:
                logger.warning(f"Reliability check failed: {e}")
        
        # Memory optimization
        if memory_optimization and len(st.session_state.analysis_history) % 3 == 0:
            try:
                memory_optimizer.cleanup_memory()
            except Exception as e:
                logger.warning(f"Memory optimization failed: {e}")
        
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")
        
        # Update analysis record
        analysis_record = {
            **analysis,
            "response": response_content,
            "sources": sources,
            "reasoning": reasoning_content,
            "tool_logs": tool_logs,
            "reliability": reliability_report,
            "status": "completed",
            "completion_time": datetime.now()
        }
        st.session_state.analysis_history.append(analysis_record)
        st.session_state.current_analysis["status"] = "completed"
        
        # Display results with enhanced formatting
        with results_container:
            st.markdown("### 📋 Executive Summary")
            
            if response_content:
                # Structure the response better
                sections = response_content.split('\n\n')
                for i, section in enumerate(sections):
                    if section.strip():
                        if i == 0:
                            st.markdown(f"**{section}**")
                        else:
                            st.markdown(section)
                
                # Extract and display key metrics
                if any(keyword in response_content.lower() for keyword in ["$", "billion", "million", "percent", "%"]):
                    st.markdown("### 📊 Key Metrics")
                    
                    # Simple metric extraction
                    import re
                    metric_patterns = [
                        r'\$[\d,.]+ (?:billion|million|thousand)',
                        r'[\d.]+% (?:growth|increase|decrease)',
                        r'[\d,]+ (?:employees|customers|users)'
                    ]
                    
                    metrics_found = []
                    for pattern in metric_patterns:
                        matches = re.findall(pattern, response_content, re.IGNORECASE)
                        metrics_found.extend(matches)
                    
                    if metrics_found:
                        cols = st.columns(min(len(metrics_found), 4))
                        for i, metric in enumerate(metrics_found[:4]):
                            with cols[i]:
                                st.metric("Key Metric", metric)
            
            else:
                st.warning("No analysis content generated. Please try again with a different query.")
        
        # Display reasoning traces
        if reasoning_content and reasoning_enabled:
            with reasoning_container:
                with st.expander("🧠 AI Reasoning Process", expanded=False):
                    st.markdown("**Chain of Thought:**")
                    st.text(reasoning_content)
        
        # Display sources and tools used
        if sources or tool_logs:
            with sources_container:
                col1, col2 = st.columns(2)
                
                with col1:
                    if sources:
                        st.markdown("### 📚 Data Sources")
                        for i, source in enumerate(sources[:5]):
                            with st.expander(f"Source {i+1}: {source.get('title', 'Unknown')[:50]}..."):
                                st.write(f"**URL:** {source.get('url', 'N/A')}")
                                st.write(f"**Relevance:** {source.get('relevance_score', 0):.2f}/1.0")
                                if source.get('snippet'):
                                    st.write(f"**Preview:** {source['snippet'][:200]}...")
                
                with col2:
                    if tool_logs:
                        st.markdown("### 🛠️ Analysis Tools Used")
                        with st.expander(f"{len(tool_logs)} tools executed"):
                            for log in tool_logs[-5:]:  # Show last 5 tool calls
                                st.text(f"⚡ {log['timestamp'][:19]}: {log['event'][:100]}...")
        
        # Display reliability report
        if reliability_report:
            with st.expander("🔍 Information Reliability Report"):
                confidence = reliability_report.get("confidence_metrics", {}).get("overall_confidence", 0.5)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Confidence", f"{confidence:.1%}")
                with col2:
                    source_count = len(reliability_report.get("source_analysis", []))
                    st.metric("Sources Verified", source_count)
                with col3:
                    consistency = reliability_report.get("confidence_metrics", {}).get("consistency_score", 0.5)
                    st.metric("Consistency Score", f"{consistency:.1%}")
        
        # Log successful completion
        bi_logger.info(f"Analysis completed successfully: {analysis['type']}")
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Analysis failed: {error_msg}")
        logger.error(f"Analysis failed: {e}\n{traceback.format_exc()}")
        
        # Update analysis status
        st.session_state.current_analysis["status"] = "failed"
        st.session_state.current_analysis["error"] = error_msg
        
        # Show fallback options
        st.markdown("### 🔄 Troubleshooting")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Retry Analysis"):
                st.session_state.current_analysis["status"] = "running"
                st.rerun()
        
        with col2:
            if st.button("🏠 Use Local Backend"):
                st.session_state.current_analysis["backend"] = "ollama"
                st.session_state.current_analysis["status"] = "running"
                st.rerun()
        
        with col3:
            if st.button("💡 Simplify Query"):
                st.info("💡 Try:\n• Shorter, more specific questions\n• Focus on one company/topic\n• Use simpler language")

# Analysis History
if st.session_state.analysis_history:
    st.markdown("---")
    st.header("📜 Analysis History")
    
    # Filter and sort options
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        history_filter = st.selectbox("Filter by Type", ["All"] + [a["type"] for a in st.session_state.analysis_history])
    with col2:
        sort_order = st.selectbox("Sort by", ["Most Recent", "Oldest First", "By Type"])
    with col3:
        show_count = st.selectbox("Show", [5, 10, 20, "All"])
    
    # Apply filters
    filtered_history = st.session_state.analysis_history
    if history_filter != "All":
        filtered_history = [a for a in filtered_history if a["type"] == history_filter]
    
    # Apply sorting
    if sort_order == "Most Recent":
        filtered_history = sorted(filtered_history, key=lambda x: x["timestamp"], reverse=True)
    elif sort_order == "Oldest First":
        filtered_history = sorted(filtered_history, key=lambda x: x["timestamp"])
    else:  # By Type
        filtered_history = sorted(filtered_history, key=lambda x: x["type"])
    
    # Apply count limit
    if show_count != "All":
        filtered_history = filtered_history[:show_count]
    
    # Display history
    for i, record in enumerate(filtered_history):
        with st.expander(f"📊 {record['type']} - {record['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Query:** {record['query'][:200]}...")
                st.write(f"**Backend:** {record['backend']} | **Model:** {record['model']}")
                
                if record.get("response"):
                    st.write(f"**Summary:** {record['response'][:300]}...")
                
                if record.get("sources"):
                    st.write(f"**Sources:** {len(record['sources'])} sources verified")
            
            with col2:
                st.write(f"**Status:** {record.get('status', 'unknown').title()}")
                if record.get("completion_time"):
                    duration = record["completion_time"] - record["timestamp"]
                    st.write(f"**Duration:** {duration.total_seconds():.1f}s")
                
                # Action buttons
                if st.button(f"🔄 Re-run", key=f"rerun_{i}"):
                    st.session_state.current_analysis = {
                        "query": record["query"],
                        "type": record["type"],
                        "backend": record["backend"],
                        "model": record["model"],
                        "timestamp": datetime.now(),
                        "status": "running"
                    }
                    st.rerun()

# Footer with system information
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**🚀 Platform Status**")
    if st.session_state.graph:
        st.success("System Operational")
    else:
        st.error("System Issues")

with col2:
    st.markdown("**🔧 Configuration**")
    st.info(f"Backend: {selected_backend.upper()}")
    st.info(f"Model: {selected_model}")

with col3:
    st.markdown("**📊 Session Stats**")
    if st.session_state.analysis_history:
        success_rate = len([a for a in st.session_state.analysis_history if a.get("status") == "completed"]) / len(st.session_state.analysis_history)
        st.metric("Success Rate", f"{success_rate:.1%}")
    else:
        st.metric("Analyses", "0")

with col4:
    st.markdown("**💾 System Info**")
    if st.session_state.system_health.get("components", {}).get("memory_optimizer"):
        mem_info = st.session_state.system_health["components"]["memory_optimizer"]
        st.metric("Memory Usage", f"{mem_info.get('memory_mb', 0):.0f} MB")
    else:
        st.metric("Memory", "Unknown")

# Real-time monitoring dashboard (optional)
if st.checkbox("📈 Show Real-time Monitoring Dashboard", value=False):
    st.markdown("### 📈 System Performance Dashboard")
    
    # Create mock real-time data for demonstration
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Requests/min", "23", "+8%")
    with col2:
        st.metric("Avg Response Time", "2.4s", "-0.6s")
    with col3:
        st.metric("Success Rate", "94.2%", "+2.1%")
    with col4:
        st.metric("Active Sessions", "7", "+3")
    
    # Performance chart
    chart_data = {
        'Time': [datetime.now() - timedelta(minutes=x) for x in range(10, 0, -1)],
        'Response_Time': [2.1, 1.8, 2.5, 1.9, 2.4, 2.0, 2.8, 1.7, 2.2, 2.4],
        'Success_Rate': [92, 95, 89, 96, 94, 97, 91, 98, 93, 94]
    }
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data['Time'],
        y=chart_data['Response_Time'],
        mode='lines+markers',
        name='Response Time (s)',
        line=dict(color='#667eea')
    ))
    
    fig.update_layout(
        title="System Performance Trends",
        xaxis_title="Time",
        yaxis_title="Response Time (seconds)",
        height=300,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Hidden debug information
if st.secrets.get("debug_mode", False):
    with st.expander("🔧 Debug Information"):
        st.json({
            "session_state_keys": list(st.session_state.keys()),
            "settings": {
                "llm_backend": settings.llm_backend,
                "thread_id": settings.thread_id,
                "has_groq_key": bool(settings.groq_api_key),
                "has_deepseek_key": bool(settings.deepseek_api_key)
            },
            "system_health": st.session_state.system_health
        })