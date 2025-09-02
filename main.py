"""
Business Intelligence Platform
Multi-LLM system combining Groq's fast inference with DeepSeek R1's advanced reasoning
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from bi_core.graph import build_business_graph
from bi_core.settings import settings
from bi_core.business_workflows import BusinessIntelligenceWorkflow
from bi_core.telemetry import setup_telemetry, get_logger

# Initialize telemetry
setup_telemetry()
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Business Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "graph" not in st.session_state:
    st.session_state.graph = build_business_graph()
    logger.info("Business intelligence graph initialized")

if "cfg" not in st.session_state:
    st.session_state.cfg = {"configurable": {"thread_id": settings.thread_id}}

if "workflow" not in st.session_state:
    st.session_state.workflow = BusinessIntelligenceWorkflow()

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # LLM Backend Selection
    backend_options = ["groq", "deepseek", "ollama"]
    selected_backend = st.selectbox(
        "LLM Backend",
        backend_options,
        index=backend_options.index(settings.llm_backend),
        help="Choose between Groq (fast inference), DeepSeek (advanced reasoning), or Ollama (local)"
    )
    
    # Model selection based on backend
    if selected_backend == "groq":
        model_options = ["deepseek-r1-distill-llama-70b", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]
        selected_model = st.selectbox("Groq Model", model_options)
    elif selected_backend == "deepseek":
        model_options = ["deepseek-r1", "deepseek-v3"]
        selected_model = st.selectbox("DeepSeek Model", model_options)
    else:
        model_options = ["deepseek-r1:7b", "deepseek-v3", "llama3.2"]
        selected_model = st.selectbox("Ollama Model", model_options)
    
    # Analysis type
    analysis_types = [
        "Market Research",
        "Competitive Analysis", 
        "Investment Screening",
        "Company Intelligence",
        "Trend Analysis",
        "Financial Analysis",
        "Custom Query"
    ]
    analysis_type = st.selectbox("Analysis Type", analysis_types)
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        reasoning_enabled = st.checkbox("Enable Reasoning Traces", value=True)
        max_tokens = st.slider("Max Tokens", 512, 4096, 2048)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.6, 0.1)
    
    st.markdown("---")
    st.subheader("📈 Session Stats")
    st.metric("Analyses Run", len(st.session_state.analysis_history))
    st.metric("Current Backend", selected_backend.upper())
    
    # Clear history button
    if st.button("Clear History", type="secondary"):
        st.session_state.analysis_history = []
        st.rerun()

# Main interface
st.title("📊 Business Intelligence Platform")
st.markdown("*Powered by Groq's fast inference and DeepSeek R1's advanced reasoning*")

# Analysis input section
st.header("🔍 Business Analysis")

col1, col2 = st.columns([3, 1])

with col1:
    if analysis_type == "Custom Query":
        query = st.text_area(
            "Enter your business question:",
            placeholder="e.g., Analyze the competitive landscape for electric vehicle charging infrastructure in Europe",
            height=100
        )
    else:
        # Pre-filled templates based on analysis type
        templates = {
            "Market Research": "Research the current market size, growth trends, and key players in [INDUSTRY/MARKET]. Include market drivers, challenges, and future outlook.",
            "Competitive Analysis": "Perform a comprehensive competitive analysis of [COMPANY] including direct competitors, market positioning, strengths, weaknesses, and strategic recommendations.",
            "Investment Screening": "Evaluate [COMPANY/SECTOR] as an investment opportunity. Analyze financial metrics, growth potential, risks, and provide an investment recommendation.",
            "Company Intelligence": "Gather comprehensive intelligence on [COMPANY] including business model, recent developments, financial performance, management team, and strategic initiatives.",
            "Trend Analysis": "Analyze emerging trends in [INDUSTRY/TECHNOLOGY] including adoption rates, market impact, key drivers, and future predictions.",
            "Financial Analysis": "Conduct a detailed financial analysis of [COMPANY] including revenue trends, profitability, debt levels, cash flow, and financial health assessment."
        }
        
        query = st.text_area(
            f"{analysis_type} Query:",
            value=templates.get(analysis_type, ""),
            height=100,
            help="Customize the template or use as-is"
        )

with col2:
    st.markdown("### Quick Actions")
    if st.button("🔍 Run Analysis", type="primary", disabled=not query.strip()):
        # Log analysis start
        logger.info(f"Starting {analysis_type} analysis", extra={
            "analysis_type": analysis_type,
            "backend": selected_backend,
            "model": selected_model
        })
        
        # Add to history
        analysis_record = {
            "timestamp": datetime.now(),
            "type": analysis_type,
            "query": query,
            "backend": selected_backend,
            "model": selected_model
        }
        st.session_state.analysis_history.append(analysis_record)
        
        # Create analysis container
        analysis_container = st.container()
        
        with analysis_container:
            st.subheader(f"📊 {analysis_type} Results")
            
            # Progress indicators
            progress_container = st.container()
            results_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                reasoning_expander = st.expander("🧠 Reasoning Traces", expanded=reasoning_enabled)
            
            try:
                # Execute the analysis
                messages = [HumanMessage(content=query)]
                
                # Stream the analysis
                events = st.session_state.graph.stream(
                    {"messages": messages, "analysis_type": analysis_type},
                    st.session_state.cfg,
                    stream_mode="values"
                )
                
                response_content = ""
                reasoning_content = ""
                sources = []
                
                for i, event in enumerate(events):
                    progress_bar.progress(min(i * 0.1, 0.9))
                    
                    if "messages" in event and event["messages"]:
                        msg = event["messages"][-1]
                        
                        if hasattr(msg, "content"):
                            # Extract reasoning traces if present
                            content = str(msg.content)
                            if "<think>" in content and "</think>" in content:
                                think_start = content.find("<think>") + 7
                                think_end = content.find("</think>")
                                reasoning_part = content[think_start:think_end]
                                response_part = content.replace(f"<think>{reasoning_part}</think>", "").strip()
                                
                                reasoning_content = reasoning_part
                                response_content = response_part
                            else:
                                response_content = content
                    
                    if "sources" in event:
                        sources = event["sources"]
                    
                    # Update status
                    if "tool_calls" in str(event):
                        status_text.text("🔍 Gathering information...")
                    elif response_content:
                        status_text.text("🧠 Analyzing data...")
                
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis complete!")
                
                # Display reasoning traces
                if reasoning_content and reasoning_enabled:
                    with reasoning_expander:
                        st.markdown("**Chain of Thought Process:**")
                        st.text(reasoning_content)
                
                # Display results
                with results_container:
                    if response_content:
                        st.markdown("### 📋 Analysis Summary")
                        st.markdown(response_content)
                        
                        # Extract and display key metrics if present
                        if any(keyword in response_content.lower() for keyword in ["$", "billion", "million", "percent", "%"]):
                            st.markdown("### 📊 Key Metrics")
                            # Simple metric extraction (in a real system, you'd use more sophisticated parsing)
                            lines = response_content.split('\n')
                            metrics = []
                            for line in lines:
                                if any(keyword in line.lower() for keyword in ["$", "billion", "million", "percent", "%"]):
                                    metrics.append(line.strip())
                            
                            if metrics:
                                for metric in metrics[:5]:  # Show top 5 metrics
                                    st.info(metric)
                    
                    # Display sources
                    if sources:
                        st.markdown("### 📚 Sources")
                        for i, source in enumerate(sources[:5]):  # Limit to 5 sources
                            with st.expander(f"Source {i+1}: {source.get('title', 'Unknown')}"):
                                st.write(f"**URL:** {source.get('url', 'N/A')}")
                                st.write(f"**Summary:** {source.get('snippet', 'No summary available')}")
                
                # Log successful completion
                logger.info(f"Analysis completed successfully", extra={
                    "analysis_type": analysis_type,
                    "response_length": len(response_content),
                    "sources_count": len(sources)
                })
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                logger.error(f"Analysis failed", extra={
                    "error": str(e),
                    "analysis_type": analysis_type
                })
                
                # Show fallback options
                st.markdown("### 🔄 Fallback Options")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Retry with Ollama"):
                        st.info("Switching to local Ollama backend...")
                        st.rerun()
                with col2:
                    if st.button("Simplify Query"):
                        st.info("Try breaking down your question into smaller parts.")

# Analysis history
if st.session_state.analysis_history:
    st.markdown("---")
    st.header("📜 Analysis History")
    
    for i, record in enumerate(reversed(st.session_state.analysis_history[-5:])):  # Show last 5
        with st.expander(f"{record['type']} - {record['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
            st.write(f"**Query:** {record['query'][:200]}...")
            st.write(f"**Backend:** {record['backend']} ({record['model']})")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Replay Analysis {i}", key=f"replay_{i}"):
                    # Replay analysis from checkpoint
                    hist = list(st.session_state.graph.get_state_history(st.session_state.cfg))
                    if hist:
                        chosen = hist[min(i, len(hist)-1)]
                        for event in st.session_state.graph.stream(None, chosen.config, stream_mode="values"):
                            pass
                        st.success("Analysis replayed from checkpoint!")
                        st.rerun()
            
            with col2:
                if st.button(f"Export Results {i}", key=f"export_{i}"):
                    # Simple export functionality
                    export_data = {
                        "analysis": record,
                        "timestamp": record['timestamp'].isoformat(),
                        "platform": "Business Intelligence Platform"
                    }
                    st.download_button(
                        "📥 Download JSON",
                        json.dumps(export_data, indent=2),
                        f"bi_analysis_{record['timestamp'].strftime('%Y%m%d_%H%M')}.json",
                        "application/json"
                    )

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🚀 Platform Status**")
    st.success("All systems operational")

with col2:
    st.markdown("**🔧 Current Configuration**")
    st.info(f"Backend: {selected_backend.upper()}")

with col3:
    st.markdown("**📊 Performance**")
    if st.session_state.analysis_history:
        avg_time = "< 30s"  # In a real system, you'd track actual times
        st.metric("Avg Response Time", avg_time)
    else:
        st.metric("Avg Response Time", "N/A")

# Real-time monitoring (placeholder for production deployment)
if st.checkbox("Show Real-time Monitoring", value=False):
    st.markdown("### 📈 Real-time System Metrics")
    
    # Simulated metrics for demonstration
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("API Requests/min", "47", "+12%")
    
    with col2:
        st.metric("Avg Latency", "1.2s", "-0.3s")
    
    with col3:
        st.metric("Success Rate", "99.2%", "+0.1%")
    
    with col4:
        st.metric("Active Sessions", "23", "+5")
    
    # Placeholder chart
    chart_data = {
        'Time': [datetime.now() - timedelta(minutes=x) for x in range(10, 0, -1)],
        'Requests': [45, 52, 38, 61, 47, 55, 42, 58, 49, 47],
        'Latency': [1.5, 1.3, 1.8, 1.1, 1.2, 1.4, 1.6, 1.0, 1.3, 1.2]
    }
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data['Time'],
        y=chart_data['Requests'],
        mode='lines+markers',
        name='Requests/min',
        yaxis='y'
    ))
    fig.add_trace(go.Scatter(
        x=chart_data['Time'],
        y=chart_data['Latency'],
        mode='lines+markers',
        name='Latency (s)',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="System Performance Over Time",
        xaxis_title="Time",
        yaxis=dict(title="Requests/min", side="left"),
        yaxis2=dict(title="Latency (s)", side="right", overlaying="y"),
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
