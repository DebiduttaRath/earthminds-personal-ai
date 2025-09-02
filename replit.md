# Business Intelligence Platform

## Overview

This is a comprehensive Business Intelligence Platform that combines multiple Large Language Models (LLMs) to provide sophisticated business analysis capabilities. The platform leverages Groq's fast inference for quick responses and DeepSeek R1's advanced reasoning for complex analytical tasks. Built as a multi-agent system using LangGraph, it orchestrates various business intelligence workflows including market research, competitive analysis, investment screening, company intelligence, trend analysis, and financial analysis.

The platform provides a Streamlit-based web interface with real-time streaming responses, source citations, tool execution traces, and checkpoint-based conversation replay. It integrates multiple data sources including Wikipedia, DuckDuckGo search, and web scraping capabilities to gather comprehensive business intelligence.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Main user interface built with Streamlit providing real-time interaction, streaming responses, and visualization capabilities
- **Interactive Dashboard**: Features configuration sidebar, analysis history, source panels, and tool execution traces
- **State Management**: Session-based state management for conversation continuity and workflow persistence

### Backend Architecture
- **LangGraph Orchestration**: State-based workflow engine managing complex multi-step business analysis processes
- **Multi-Agent System**: Specialized agents for different analysis types (market research, competitive analysis, financial analysis, etc.)
- **LLM Factory Pattern**: Intelligent routing between different LLM backends with automatic fallback mechanisms
- **Workflow Engine**: Pre-defined business intelligence workflows with complexity assessment and dynamic routing

### LLM Integration
- **Primary Backend (Groq)**: Fast inference using Llama-3.1-70B and Mixtral-8x7B models for quick responses
- **Reasoning Backend (DeepSeek)**: Advanced reasoning capabilities using DeepSeek R1 and V3 models
- **Local Fallback (Ollama)**: Offline capabilities with local model deployment for resilience
- **Intelligent Routing**: Automatic selection between fast inference and deep reasoning based on query complexity

### Data Architecture
- **SQLite Checkpointing**: Conversation state persistence and replay capabilities
- **Multi-Source Data Integration**: Wikipedia API, DuckDuckGo search, web scraping with trafilatura
- **Structured Data Extraction**: Financial metrics, company information, and market data parsing
- **Citation Management**: Source tracking with titles, URLs, and relevance scoring

### Tool System
- **Business Intelligence Tools**: Specialized tools for wiki search, web search, content fetching, financial analysis, company news, and market data
- **Web Scraping**: Advanced content extraction with readability processing
- **Financial Calculator**: Business-focused calculations with formatting for financial metrics
- **Rate Limiting**: HTTP request management with retries and backpressure handling

### Observability & Telemetry
- **OpenTelemetry Integration**: Distributed tracing and metrics collection
- **Structured Logging**: Comprehensive logging with business context
- **Performance Monitoring**: Request timing, model performance, and tool execution tracking
- **Error Handling**: Graceful fallbacks with detailed error reporting

### Configuration Management
- **Environment-based Configuration**: Flexible settings management through environment variables
- **Multi-Backend Support**: Configuration for Groq, DeepSeek, and Ollama backends
- **Runtime Parameters**: Adjustable context lengths, temperatures, and timeout settings
- **Feature Flags**: Toggleable features for reasoning traces, financial data, and competitor analysis

## External Dependencies

### LLM Services
- **Groq API**: Primary LLM backend for fast inference with Llama-3.1-70B and Mixtral-8x7B models
- **DeepSeek API**: Advanced reasoning capabilities with R1 and V3 model variants
- **Ollama**: Local LLM deployment for offline capabilities and fallback scenarios

### Data Sources & APIs
- **Wikipedia API (MediaWiki)**: Comprehensive business and company information retrieval
- **DuckDuckGo Search**: Privacy-focused web search for current business intelligence
- **Web Scraping (trafilatura)**: Clean text extraction from business websites and news sources

### Core Dependencies
- **LangChain/LangGraph**: Agent orchestration, tool management, and conversation state handling
- **Streamlit**: Web application framework for interactive business intelligence dashboard
- **Plotly**: Advanced data visualization and chart generation for business metrics
- **SQLite**: Local database for conversation checkpoints and state persistence

### Python Libraries
- **Requests**: HTTP client with business-appropriate headers and session management
- **BeautifulSoup**: HTML parsing for web content extraction
- **Pydantic**: Data validation and settings management with type safety
- **OpenTelemetry**: Observability stack for monitoring and performance tracking
- **Tenacity**: Retry logic with exponential backoff for external API calls

### Development & Testing
- **pytest**: Comprehensive testing framework for tools, workflows, and integrations
- **mypy**: Static type checking for code reliability
- **ruff**: Code linting and formatting for maintainable codebase

### Optional Integrations
- **PostgreSQL**: Alternative database backend (can be added via Drizzle if needed)
- **FastAPI**: Optional API service layer for backend separation
- **DuckDB**: Alternative analytical database for complex queries