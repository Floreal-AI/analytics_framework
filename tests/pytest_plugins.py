"""
Custom pytest plugins for the Bittensor analytics framework.
"""

import os
import json
import pytest
import datetime
from pathlib import Path
from _pytest.runner import pytest_runtest_protocol
from _pytest.terminal import TerminalReporter

from conversion_subnet.utils.log import logger
from tests.test_utils import ensure_test_results_dir, save_test_logs


def pytest_configure(config):
    """Configure pytest for the analytics framework."""
    # Ensure the test results directory exists
    ensure_test_results_dir()
    
    # Set up the logger
    test_results_dir = ensure_test_results_dir()
    logs_dir = test_results_dir / "logs"
    logger.setup(path=logs_dir, level="DEBUG")
    
    # Register an additional marker
    config.addinivalue_line(
        "markers", "analytics: tests related to analytics framework"
    )


def pytest_unconfigure(config):
    """Actions to run after pytest finishes."""
    # Save the test logs
    save_test_logs()
    
    # Log test session summary
    terminal_reporter = config.pluginmanager.get_plugin('terminalreporter')
    if terminal_reporter:
        stats = terminal_reporter.stats
        logger.info(
            "Test session summary", 
            passed=len(stats.get('passed', [])),
            failed=len(stats.get('failed', [])),
            skipped=len(stats.get('skipped', [])),
            error=len(stats.get('error', []))
        )


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    # Save results summary
    test_results_dir = ensure_test_results_dir()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get stats from terminal reporter
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    
    # Get session start time (handle different pytest versions)
    session_start_time = None
    if hasattr(reporter, '_sessionstarttime'):
        session_start_time = reporter._sessionstarttime
    elif hasattr(reporter, '_session_start'):
        session_start_time = reporter._session_start
    elif hasattr(session, 'testsfailed'):
        # Fallback: session may have started timestamp
        session_start_time = getattr(session, '_start_time', None)
    
    # Calculate duration - handle different timestamp types
    duration = None
    if session_start_time is not None:
        try:
            # Convert to float if it's not already
            if hasattr(session_start_time, 'timestamp'):
                # It's a datetime-like object
                start_timestamp = session_start_time.timestamp()
            elif isinstance(session_start_time, (int, float)):
                # It's already a timestamp
                start_timestamp = float(session_start_time)
            else:
                # Try to convert to float
                start_timestamp = float(session_start_time)
            
            duration = datetime.datetime.now().timestamp() - start_timestamp
        except (AttributeError, TypeError, ValueError):
            # If conversion fails, just set duration to None
            duration = None
    
    # Create the test summary
    summary = {
        "exitstatus": exitstatus,
        "test_counts": {
            "total": getattr(reporter, '_numcollected', 0),
            "passed": len(reporter.stats.get("passed", [])),
            "failed": len(reporter.stats.get("failed", [])),
            "skipped": len(reporter.stats.get("skipped", [])),
            "errors": len(reporter.stats.get("error", [])),
        },
        "duration": duration
    }
    
    # Add detailed test results if available
    if hasattr(reporter, "stats"):
        failed_tests = []
        for test in reporter.stats.get("failed", []):
            failed_tests.append({
                "name": test.nodeid,
                "duration": getattr(test, "duration", None),
                "error_message": str(getattr(test, "longrepr", ""))
            })
        
        if failed_tests:
            summary["failed_tests"] = failed_tests
    
    # Save the summary
    summary_path = test_results_dir / "reports" / f"summary_{timestamp}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Log the summary
    logger.info(f"Test session finished with exit status {exitstatus}", **summary) 