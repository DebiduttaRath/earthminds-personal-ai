"""
Telemetry and Observability for Business Intelligence Platform
OpenTelemetry implementation for monitoring and logging
"""

import logging
import os
import time
from typing import Dict, Any, Optional
from functools import wraps
from contextlib import contextmanager

try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    metrics = None

from bi_core.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('business_intelligence.log') if os.path.exists('.') else logging.NullHandler()
    ]
)

class TelemetryManager:
    """Centralized telemetry management"""
    
    def __init__(self):
        self._tracer = None
        self._meter = None
        self._initialized = False
        
        # Metrics
        self._request_counter = None
        self._request_duration = None
        self._error_counter = None
        self._llm_token_counter = None
        
        # Initialize if OpenTelemetry is available and configured
        if OTEL_AVAILABLE and settings.otel_endpoint:
            self._initialize_otel()
    
    def _initialize_otel(self):
        """Initialize OpenTelemetry tracing and metrics"""
        try:
            # Create resource
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: settings.otel_service_name,
                ResourceAttributes.SERVICE_VERSION: "1.0.0",
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: os.getenv("ENVIRONMENT", "development")
            })
            
            # Initialize tracing
            trace_exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint)
            trace_processor = BatchSpanProcessor(trace_exporter)
            tracer_provider = TracerProvider(resource=resource)
            tracer_provider.add_span_processor(trace_processor)
            trace.set_tracer_provider(tracer_provider)
            
            self._tracer = trace.get_tracer(__name__)
            
            # Initialize metrics
            metric_exporter = OTLPMetricExporter(endpoint=settings.otel_endpoint)
            metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=10000)
            meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            metrics.set_meter_provider(meter_provider)
            
            self._meter = metrics.get_meter(__name__)
            
            # Create metrics
            self._request_counter = self._meter.create_counter(
                name="bi_requests_total",
                description="Total number of business intelligence requests",
                unit="1"
            )
            
            self._request_duration = self._meter.create_histogram(
                name="bi_request_duration_seconds",
                description="Duration of business intelligence requests",
                unit="s"
            )
            
            self._error_counter = self._meter.create_counter(
                name="bi_errors_total",
                description="Total number of errors in business intelligence operations",
                unit="1"
            )
            
            self._llm_token_counter = self._meter.create_counter(
                name="bi_llm_tokens_total",
                description="Total number of LLM tokens consumed",
                unit="1"
            )
            
            self._initialized = True
            logging.info("OpenTelemetry initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize OpenTelemetry: {e}")
            self._initialized = False
    
    def get_tracer(self):
        """Get the tracer instance"""
        return self._tracer if self._initialized else None
    
    def get_meter(self):
        """Get the meter instance"""
        return self._meter if self._initialized else None
    
    def record_request(self, analysis_type: str, duration: float, success: bool = True):
        """Record a business intelligence request"""
        if not self._initialized:
            return
        
        try:
            # Record request
            self._request_counter.add(1, {
                "analysis_type": analysis_type,
                "status": "success" if success else "error"
            })
            
            # Record duration
            self._request_duration.record(duration, {
                "analysis_type": analysis_type
            })
            
            if not success:
                self._error_counter.add(1, {"analysis_type": analysis_type})
                
        except Exception as e:
            logging.error(f"Failed to record request metrics: {e}")
    
    def record_llm_usage(self, backend: str, model: str, input_tokens: int, output_tokens: int):
        """Record LLM token usage"""
        if not self._initialized:
            return
        
        try:
            self._llm_token_counter.add(input_tokens, {
                "backend": backend,
                "model": model,
                "token_type": "input"
            })
            
            self._llm_token_counter.add(output_tokens, {
                "backend": backend,
                "model": model,
                "token_type": "output"
            })
            
        except Exception as e:
            logging.error(f"Failed to record LLM usage: {e}")
    
    @contextmanager
    def trace_operation(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager for tracing operations"""
        if not self._initialized or not self._tracer:
            yield None
            return
        
        with self._tracer.start_as_current_span(operation_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            
            start_time = time.time()
            try:
                yield span
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration_seconds", duration)

# Global telemetry manager
_telemetry_manager = TelemetryManager()

def setup_telemetry():
    """Setup telemetry (called from main application)"""
    global _telemetry_manager
    if not _telemetry_manager._initialized and OTEL_AVAILABLE and settings.otel_endpoint:
        _telemetry_manager._initialize_otel()

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with proper configuration"""
    logger = logging.getLogger(name)
    
    # Add custom handler for structured logging if needed
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

def trace_business_operation(operation_name: str):
    """Decorator for tracing business operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attributes = {
                "function_name": func.__name__,
                "module": func.__module__
            }
            
            # Add relevant arguments to attributes
            if args and hasattr(args[0], '__class__'):
                attributes["class_name"] = args[0].__class__.__name__
            
            with _telemetry_manager.trace_operation(operation_name, attributes):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Record success metric
                    _telemetry_manager.record_request(
                        analysis_type=operation_name,
                        duration=duration,
                        success=True
                    )
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    
                    # Record error metric
                    _telemetry_manager.record_request(
                        analysis_type=operation_name,
                        duration=duration,
                        success=False
                    )
                    
                    raise
                    
        return wrapper
    return decorator

def record_llm_metrics(backend: str, model: str, input_tokens: int = 0, output_tokens: int = 0):
    """Record LLM usage metrics"""
    _telemetry_manager.record_llm_usage(backend, model, input_tokens, output_tokens)

class BusinessMetrics:
    """Business-specific metrics tracking"""
    
    def __init__(self):
        self.analysis_counts = {}
        self.error_counts = {}
        self.average_durations = {}
        self.llm_usage = {}
    
    def record_analysis(self, analysis_type: str, duration: float, success: bool = True):
        """Record analysis execution"""
        # Update counts
        if analysis_type not in self.analysis_counts:
            self.analysis_counts[analysis_type] = {"success": 0, "error": 0}
        
        if success:
            self.analysis_counts[analysis_type]["success"] += 1
        else:
            self.analysis_counts[analysis_type]["error"] += 1
        
        # Update average durations
        if analysis_type not in self.average_durations:
            self.average_durations[analysis_type] = {"total": 0, "count": 0}
        
        self.average_durations[analysis_type]["total"] += duration
        self.average_durations[analysis_type]["count"] += 1
        
        # Log to OpenTelemetry if available
        _telemetry_manager.record_request(analysis_type, duration, success)
    
    def record_llm_call(self, backend: str, model: str, tokens_used: int):
        """Record LLM API call"""
        key = f"{backend}:{model}"
        if key not in self.llm_usage:
            self.llm_usage[key] = {"calls": 0, "tokens": 0}
        
        self.llm_usage[key]["calls"] += 1
        self.llm_usage[key]["tokens"] += tokens_used
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {
            "analysis_counts": self.analysis_counts,
            "error_rates": {},
            "average_durations": {},
            "llm_usage": self.llm_usage
        }
        
        # Calculate error rates
        for analysis_type, counts in self.analysis_counts.items():
            total = counts["success"] + counts["error"]
            if total > 0:
                summary["error_rates"][analysis_type] = counts["error"] / total
        
        # Calculate average durations
        for analysis_type, duration_data in self.average_durations.items():
            if duration_data["count"] > 0:
                summary["average_durations"][analysis_type] = (
                    duration_data["total"] / duration_data["count"]
                )
        
        return summary

# Global metrics instance
business_metrics = BusinessMetrics()

# Export main functions and classes
__all__ = [
    "setup_telemetry",
    "get_logger", 
    "trace_business_operation",
    "record_llm_metrics",
    "business_metrics",
    "TelemetryManager"
]
