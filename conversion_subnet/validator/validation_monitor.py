"""
Validation Monitoring System

This module provides comprehensive monitoring, alerting, and health checking
for the external validation API. Enables proactive monitoring of validation
reliability without creating fallback implementations.
"""

import time
import json
import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import statistics
from collections import deque, defaultdict

from conversion_subnet.utils.log import logger
from conversion_subnet.validator.validation_client import ValidationAPIClient, ValidationError, CircuitBreakerError

@dataclass
class ValidationAlert:
    """Represents a validation alert."""
    id: str
    severity: str  # "info", "warning", "error", "critical"
    title: str
    message: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_timestamp: Optional[float] = None

@dataclass
class HealthCheckResult:
    """Result from a health check."""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: float
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

class ValidationMonitor:
    """
    Comprehensive monitoring system for external validation reliability.
    
    Provides health checking, alerting, metrics collection, and monitoring
    without creating fallback implementations.
    """
    
    def __init__(self, 
                 validation_client: ValidationAPIClient,
                 health_check_interval: float = 60.0,
                 metrics_retention_hours: int = 24,
                 alert_retention_hours: int = 168):  # 1 week
        """
        Initialize the validation monitor.
        
        Args:
            validation_client: The validation client to monitor
            health_check_interval: Seconds between health checks
            metrics_retention_hours: Hours to retain detailed metrics
            alert_retention_hours: Hours to retain alert history
        """
        self.validation_client = validation_client
        self.health_check_interval = health_check_interval
        self.metrics_retention_hours = metrics_retention_hours
        self.alert_retention_hours = alert_retention_hours
        
        # Monitoring state
        self._health_history: deque = deque(maxlen=1000)  # Recent health checks
        self._alerts: Dict[str, ValidationAlert] = {}
        self._alert_history: deque = deque(maxlen=10000)  # All alerts
        self._response_times: deque = deque(maxlen=1000)  # Recent response times
        self._error_counts: defaultdict = defaultdict(int)
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Alert handlers
        self._alert_handlers: List[Callable[[ValidationAlert], None]] = []
        
        # Health check test data
        self._health_check_test_pk = "health_check_test_pk_12345"
        
        logger.info(f"ValidationMonitor initialized with {health_check_interval}s health checks")
    
    def add_alert_handler(self, handler: Callable[[ValidationAlert], None]) -> None:
        """Add an alert handler function."""
        self._alert_handlers.append(handler)
        logger.info(f"Added alert handler: {handler.__name__}")
    
    def remove_alert_handler(self, handler: Callable[[ValidationAlert], None]) -> None:
        """Remove an alert handler function."""
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)
            logger.info(f"Removed alert handler: {handler.__name__}")
    
    async def start_monitoring(self) -> None:
        """Start continuous monitoring."""
        if self._monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Validation monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Validation monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Starting validation monitoring loop")
        
        while self._monitoring_active:
            try:
                # Perform health check
                health_result = await self._perform_health_check()
                self._health_history.append(health_result)
                
                # Analyze health and generate alerts
                await self._analyze_health_and_alert(health_result)
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Wait for next check
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(min(self.health_check_interval, 30.0))  # Don't spam errors
    
    async def _perform_health_check(self) -> HealthCheckResult:
        """Perform a health check of the validation API."""
        start_time = time.time()
        
        try:
            # Try to get validation result for a health check test_pk
            # This is a real API call to verify the service is working
            result = await self.validation_client.get_validation_result(self._health_check_test_pk)
            
            response_time = time.time() - start_time
            self._response_times.append(response_time)
            
            # Even if we get a valid response (even "not found"), the API is healthy
            return HealthCheckResult(
                status="healthy",
                timestamp=start_time,
                response_time=response_time,
                details={
                    "circuit_breaker_state": self.validation_client._circuit_state.value,
                    "test_pk": self._health_check_test_pk
                }
            )
            
        except CircuitBreakerError as e:
            return HealthCheckResult(
                status="unhealthy",
                timestamp=start_time,
                error_message=str(e),
                details={
                    "error_type": "circuit_breaker_open",
                    "circuit_breaker_state": self.validation_client._circuit_state.value
                }
            )
            
        except ValidationError as e:
            response_time = time.time() - start_time
            
            # Check if it's a "not found" error - that's actually healthy
            if "status 404" in str(e) or "not found" in str(e).lower():
                return HealthCheckResult(
                    status="healthy",
                    timestamp=start_time,
                    response_time=response_time,
                    details={
                        "message": "API responding (expected 404 for health check)",
                        "circuit_breaker_state": self.validation_client._circuit_state.value
                    }
                )
            else:
                self._error_counts[str(e)[:100]] += 1  # Track error types
                
                # Determine severity based on error type
                if "timeout" in str(e).lower():
                    status = "degraded"
                elif "5" in str(e) and "status" in str(e):  # 5xx errors
                    status = "degraded"
                else:
                    status = "unhealthy"
                
                return HealthCheckResult(
                    status=status,
                    timestamp=start_time,
                    response_time=response_time,
                    error_message=str(e),
                    details={
                        "error_type": type(e).__name__,
                        "attempt_number": getattr(e, 'attempt_number', 1),
                        "is_retryable": getattr(e, 'is_retryable', True),
                        "circuit_breaker_state": self.validation_client._circuit_state.value
                    }
                )
        
        except Exception as e:
            self._error_counts[str(e)[:100]] += 1
            return HealthCheckResult(
                status="unhealthy",
                timestamp=start_time,
                error_message=str(e),
                details={
                    "error_type": type(e).__name__,
                    "unexpected_error": True
                }
            )
    
    async def _analyze_health_and_alert(self, health_result: HealthCheckResult) -> None:
        """Analyze health results and generate alerts."""
        current_time = time.time()
        
        # Get recent health history for analysis
        recent_checks = [h for h in self._health_history if current_time - h.timestamp < 300]  # Last 5 minutes
        
        if not recent_checks:
            return
        
        # Calculate health metrics
        total_checks = len(recent_checks)
        healthy_checks = sum(1 for h in recent_checks if h.status == "healthy")
        degraded_checks = sum(1 for h in recent_checks if h.status == "degraded")
        unhealthy_checks = sum(1 for h in recent_checks if h.status == "unhealthy")
        
        health_rate = healthy_checks / total_checks
        
        # Circuit breaker alerts
        cb_state = self.validation_client._circuit_state.value
        if cb_state == "open":
            await self._create_alert(
                "circuit_breaker_open",
                "critical",
                "Circuit Breaker Open",
                f"Validation circuit breaker is OPEN. Service unavailable. "
                f"Failed {self.validation_client._failure_count} times.",
                {"circuit_breaker_state": cb_state, "failure_count": self.validation_client._failure_count}
            )
        elif cb_state == "half_open":
            await self._create_alert(
                "circuit_breaker_half_open",
                "warning",
                "Circuit Breaker Testing Recovery",
                "Validation circuit breaker is HALF_OPEN, testing service recovery.",
                {"circuit_breaker_state": cb_state}
            )
        else:
            # Resolve circuit breaker alerts if closed
            await self._resolve_alert("circuit_breaker_open")
            await self._resolve_alert("circuit_breaker_half_open")
        
        # Health rate alerts
        if health_rate < 0.5:  # Less than 50% healthy
            await self._create_alert(
                "low_health_rate",
                "critical",
                "Low Validation Health Rate",
                f"Only {health_rate:.1%} of recent validation checks were healthy "
                f"({healthy_checks}/{total_checks} in last 5 minutes)",
                {
                    "health_rate": health_rate,
                    "healthy_checks": healthy_checks,
                    "total_checks": total_checks,
                    "degraded_checks": degraded_checks,
                    "unhealthy_checks": unhealthy_checks
                }
            )
        elif health_rate < 0.8:  # Less than 80% healthy
            await self._create_alert(
                "degraded_health_rate",
                "warning",
                "Degraded Validation Health",
                f"Validation health rate is {health_rate:.1%} ({healthy_checks}/{total_checks} in last 5 minutes)",
                {
                    "health_rate": health_rate,
                    "healthy_checks": healthy_checks,
                    "total_checks": total_checks
                }
            )
        else:
            # Resolve health rate alerts
            await self._resolve_alert("low_health_rate")
            await self._resolve_alert("degraded_health_rate")
        
        # Response time alerts
        if self._response_times and len(self._response_times) >= 5:
            recent_response_times = list(self._response_times)[-10:]  # Last 10 measurements
            avg_response_time = statistics.mean(recent_response_times)
            
            if avg_response_time > 30.0:  # More than 30 seconds
                await self._create_alert(
                    "slow_response_time",
                    "warning",
                    "Slow Validation Response Time",
                    f"Average validation response time is {avg_response_time:.2f}s (last 10 requests)",
                    {"average_response_time": avg_response_time, "sample_size": len(recent_response_times)}
                )
            else:
                await self._resolve_alert("slow_response_time")
        
        # Error pattern alerts
        await self._analyze_error_patterns()
    
    async def _analyze_error_patterns(self) -> None:
        """Analyze error patterns and create alerts."""
        if not self._error_counts:
            return
        
        # Find most common errors
        sorted_errors = sorted(self._error_counts.items(), key=lambda x: x[1], reverse=True)
        
        for error_msg, count in sorted_errors[:3]:  # Top 3 errors
            if count >= 5:  # At least 5 occurrences
                alert_id = f"error_pattern_{hash(error_msg) % 10000}"
                await self._create_alert(
                    alert_id,
                    "warning",
                    "Repeated Validation Error",
                    f"Error pattern detected: '{error_msg}' occurred {count} times",
                    {"error_message": error_msg, "occurrence_count": count}
                )
    
    async def _create_alert(self, alert_id: str, severity: str, title: str, message: str, metadata: Dict[str, Any]) -> None:
        """Create or update an alert."""
        current_time = time.time()
        
        if alert_id in self._alerts and not self._alerts[alert_id].resolved:
            # Alert already exists and is active - just update metadata
            self._alerts[alert_id].metadata.update(metadata)
            return
        
        # Create new alert
        alert = ValidationAlert(
            id=alert_id,
            severity=severity,
            title=title,
            message=message,
            timestamp=current_time,
            metadata=metadata
        )
        
        self._alerts[alert_id] = alert
        self._alert_history.append(alert)
        
        logger.warning(f"ALERT [{severity.upper()}] {title}: {message}")
        
        # Notify alert handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler {handler.__name__}: {e}")
    
    async def _resolve_alert(self, alert_id: str) -> None:
        """Resolve an active alert."""
        if alert_id in self._alerts and not self._alerts[alert_id].resolved:
            current_time = time.time()
            self._alerts[alert_id].resolved = True
            self._alerts[alert_id].resolved_timestamp = current_time
            
            logger.info(f"RESOLVED: Alert {alert_id} - {self._alerts[alert_id].title}")
    
    def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data."""
        current_time = time.time()
        
        # Clean up alert history
        cutoff_time = current_time - (self.alert_retention_hours * 3600)
        while self._alert_history and self._alert_history[0].timestamp < cutoff_time:
            self._alert_history.popleft()
        
        # Reset error counts periodically (every hour)
        if hasattr(self, '_last_error_reset'):
            if current_time - self._last_error_reset > 3600:  # 1 hour
                self._error_counts.clear()
                self._last_error_reset = current_time
        else:
            self._last_error_reset = current_time
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        current_time = time.time()
        
        # Calculate recent metrics
        recent_checks = [h for h in self._health_history if current_time - h.timestamp < 300]  # Last 5 minutes
        
        if recent_checks:
            healthy_count = sum(1 for h in recent_checks if h.status == "healthy")
            health_rate = healthy_count / len(recent_checks)
            
            response_times = [h.response_time for h in recent_checks if h.response_time is not None]
            avg_response_time = statistics.mean(response_times) if response_times else None
        else:
            health_rate = None
            avg_response_time = None
        
        # Get active alerts
        active_alerts = [alert for alert in self._alerts.values() if not alert.resolved]
        
        # Get validation client health
        client_health = self.validation_client.get_health_status()
        
        return {
            "monitoring": {
                "active": self._monitoring_active,
                "health_check_interval": self.health_check_interval,
                "last_check": self._health_history[-1].timestamp if self._health_history else None
            },
            "health": {
                "current_status": self._health_history[-1].status if self._health_history else "unknown",
                "recent_health_rate": health_rate,
                "recent_avg_response_time": avg_response_time,
                "total_health_checks": len(self._health_history)
            },
            "alerts": {
                "active_count": len(active_alerts),
                "active_alerts": [asdict(alert) for alert in active_alerts],
                "total_alerts": len(self._alert_history)
            },
            "errors": {
                "common_errors": dict(list(sorted(self._error_counts.items(), key=lambda x: x[1], reverse=True))[:5])
            },
            "validation_client": client_health
        }
    
    def get_active_alerts(self) -> List[ValidationAlert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self._alerts.values() if not alert.resolved]
    
    def get_recent_health_history(self, hours: int = 1) -> List[HealthCheckResult]:
        """Get health check history for the specified number of hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [h for h in self._health_history if h.timestamp >= cutoff_time]
    
    async def force_health_check(self) -> HealthCheckResult:
        """Force an immediate health check."""
        logger.info("Performing forced health check")
        result = await self._perform_health_check()
        self._health_history.append(result)
        await self._analyze_health_and_alert(result)
        return result
    
    def reset_alerts(self) -> None:
        """Reset all alerts (for administrative use)."""
        logger.warning("Manually resetting all alerts")
        for alert in self._alerts.values():
            if not alert.resolved:
                alert.resolved = True
                alert.resolved_timestamp = time.time()
    
    def export_monitoring_data(self, file_path: str) -> None:
        """Export monitoring data to JSON file."""
        data = {
            "export_timestamp": time.time(),
            "health_history": [asdict(h) for h in self._health_history],
            "alert_history": [asdict(a) for a in self._alert_history],
            "error_counts": dict(self._error_counts),
            "monitoring_status": self.get_monitoring_status()
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Monitoring data exported to {file_path}")

# Default alert handlers
def log_alert_handler(alert: ValidationAlert) -> None:
    """Simple alert handler that logs alerts."""
    level_map = {
        "info": logger.info,
        "warning": logger.warning,
        "error": logger.error,
        "critical": logger.critical
    }
    
    log_func = level_map.get(alert.severity, logger.warning)
    log_func(f"VALIDATION ALERT [{alert.severity.upper()}] {alert.title}: {alert.message}")

def console_alert_handler(alert: ValidationAlert) -> None:
    """Alert handler that prints to console with formatting."""
    timestamp = datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    severity_icon = {
        "info": "ℹ️",
        "warning": "⚠️", 
        "error": "❌",
        "critical": "🚨"
    }
    
    icon = severity_icon.get(alert.severity, "⚠️")
    print(f"\n{icon} VALIDATION ALERT [{alert.severity.upper()}] - {timestamp}")
    print(f"   {alert.title}")
    print(f"   {alert.message}")
    if alert.metadata:
        print(f"   Details: {alert.metadata}")
    print()

# Global monitoring instance
_global_monitor: Optional[ValidationMonitor] = None

def get_global_monitor() -> Optional[ValidationMonitor]:
    """Get the global monitoring instance."""
    return _global_monitor

def configure_global_monitor(validation_client: ValidationAPIClient, **kwargs) -> ValidationMonitor:
    """Configure the global monitoring instance."""
    global _global_monitor
    _global_monitor = ValidationMonitor(validation_client, **kwargs)
    
    # Add default alert handlers
    _global_monitor.add_alert_handler(log_alert_handler)
    _global_monitor.add_alert_handler(console_alert_handler)
    
    logger.info("Global validation monitor configured")
    return _global_monitor 