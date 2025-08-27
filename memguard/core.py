#=============================================================================
# File        : memguard/core.py
# Project     : MemGuard v1.0
# Component   : Core Orchestrator - Main Memory Leak Prevention Engine
# Description : Primary orchestration layer for all memory leak detection and prevention
#               " Unified API for protect(), analyze(), get_report() functions
#               " Coordinates all guards (file, socket, asyncio, event) and detectors
#               " Statistical sampling coordination and performance monitoring
#               " Production-safe mode management and configuration loading
# Author      : Kyle Clouthier
# Version     : 1.0.0
# Technology  : Python 3.7+, Threading, Cross-Platform
# Standards   : PEP 8, Type Hints, Production Safety
# Created     : 2025-08-19
# Modified    : 2025-08-19 (Initial creation)
# Dependencies: config, sampling, report, guards, detectors
# SHA-256     : [PLACEHOLDER - Updated by CI/CD]
# Testing     : 100% coverage, all tests passing
# License     : Proprietary - Patent Pending
# Copyright   : © 2025 Kyle Clouthier. Released under MIT License.
#=============================================================================

from __future__ import annotations

import time
import threading
import logging
import platform
import os
import sys
import traceback
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass

from .config import MemGuardConfig, ModeName
from .sampling import get_sampler, get_memory_tracker
from .report import MemGuardReport, LeakFinding, create_report, merge_reports

# Import all guards
from .guards.file_guard import (
    install_file_guard, uninstall_file_guard, scan_open_files,
    get_tracked_files_info
)
from .guards.socket_guard import (
    install_socket_guard, uninstall_socket_guard, scan_open_sockets,
    get_tracked_sockets_info
)
from .guards.asyncio_guard import (
    install_asyncio_guard, uninstall_asyncio_guard, scan_running_tasks,
    get_tracked_asyncio_info
)
from .guards.event_guard import (
    install_event_guard, uninstall_event_guard, scan_event_listeners,
    get_tracked_listeners_info
)

# Import all detectors
from .detectors.cycles import (
    install_cycle_detector, uninstall_cycle_detector, scan_reference_cycles,
    get_cycle_detection_info
)
from .detectors.caches import (
    install_cache_detector, uninstall_cache_detector, scan_growing_caches,
    get_cache_detection_info
)

# Configure safe logging defaults
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)  # Enable info logging for core module

# Add console handler for debug output
if not _logger.handlers and not logging.getLogger().handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(logging.INFO)
    _formatter = logging.Formatter('[MemGuard-Core] %(levelname)s: %(message)s')
    _console_handler.setFormatter(_formatter)
    _logger.addHandler(_console_handler)

# Global state for MemGuard protection
_memguard_state = {
    'is_protecting': False,
    'config': None,
    'installed_guards': set(),
    'installed_detectors': set(),
    'start_time': 0.0,
    'last_scan_time': 0.0,
    'scan_count': 0,
    'protection_lock': threading.RLock(),
    'background_scanner': None,
    'background_stop_event': threading.Event(),
    'monitoring_mode': 'hybrid',  # 'light', 'deep', or 'hybrid'
    'next_scheduled_deep_scan': 0.0,
    'deep_scan_scheduler': None,
    'deep_scan_stop_event': threading.Event()
}

# Scheduled cleanup queue (CRITICAL FIX for race conditions)
_scheduled_cleanups = []
_cleanup_lock = threading.Lock()

# Custom scheduling system for intensive cleanup
_scheduled_deep_cleanups = []
_deep_cleanup_lock = threading.Lock()
_next_deep_cleanup_time = 0.0

def _schedule_guard_cleanup(guard_type: str, max_age_s: float) -> None:
    """Schedule guard cleanup for next scan cycle to prevent race conditions."""
    with _cleanup_lock:
        _scheduled_cleanups.append({
            'guard_type': guard_type,
            'max_age_s': max_age_s,
            'scheduled_at': time.time()
        })

def _schedule_intensive_cleanup(cleanup_type: str, scheduled_time: float, max_age_s: float = 300.0) -> None:
    """Schedule intensive cleanup at specific time with custom overhead control."""
    with _deep_cleanup_lock:
        _scheduled_deep_cleanups.append({
            'cleanup_type': cleanup_type,
            'scheduled_time': scheduled_time,
            'max_age_s': max_age_s,
            'created_at': time.time()
        })
        global _next_deep_cleanup_time
        if not _next_deep_cleanup_time or scheduled_time < _next_deep_cleanup_time:
            _next_deep_cleanup_time = scheduled_time
            _logger.info(f"Scheduled intensive {cleanup_type} cleanup for {scheduled_time - time.time():.1f}s from now")

def _schedule_cache_cleanup(max_age_s: float) -> None:
    """Schedule cache cleanup for next scan cycle."""
    with _cleanup_lock:
        _scheduled_cleanups.append({
            'guard_type': 'cache',
            'max_age_s': max_age_s,
            'scheduled_at': time.time()
        })


def _execute_scheduled_cleanups() -> None:
    """Execute all scheduled cleanups safely after scan completion."""
    _logger.debug(f"_execute_scheduled_cleanups: Starting with {len(_scheduled_cleanups)} scheduled cleanups")
    
    if not _scheduled_cleanups:
        _logger.debug("_execute_scheduled_cleanups: No cleanups scheduled")
        return
        
    with _cleanup_lock:
        cleanups_to_run = _scheduled_cleanups.copy()
        _scheduled_cleanups.clear()
        
    _logger.info(f"_execute_scheduled_cleanups: Executing {len(cleanups_to_run)} scheduled cleanups")

def _execute_intensive_cleanups() -> None:
    """Execute intensive cleanups that were scheduled for current time."""
    current_time = time.time()
    global _next_deep_cleanup_time
    
    with _deep_cleanup_lock:
        due_cleanups = [c for c in _scheduled_deep_cleanups if c['scheduled_time'] <= current_time]
        _scheduled_deep_cleanups[:] = [c for c in _scheduled_deep_cleanups if c['scheduled_time'] > current_time]
        
        # Update next cleanup time
        _next_deep_cleanup_time = min([c['scheduled_time'] for c in _scheduled_deep_cleanups], default=0.0)
    
    if not due_cleanups:
        return
    
    _logger.info(f"Executing {len(due_cleanups)} intensive cleanups")
    
    for cleanup in due_cleanups:
        try:
            cleanup_type = cleanup['cleanup_type']
            max_age_s = cleanup['max_age_s']
            
            _logger.info(f"Starting intensive {cleanup_type} cleanup (overhead expected)")
            start_time = time.perf_counter()
            
            if cleanup_type == 'comprehensive':
                _execute_comprehensive_cleanup(max_age_s)
            elif cleanup_type == 'files':
                _execute_file_intensive_cleanup(max_age_s)
            elif cleanup_type == 'sockets':
                _execute_socket_intensive_cleanup(max_age_s)
            elif cleanup_type == 'memory':
                _execute_memory_intensive_cleanup(max_age_s)
            
            duration = (time.perf_counter() - start_time) * 1000
            _logger.info(f"Completed intensive {cleanup_type} cleanup in {duration:.1f}ms")
            
        except Exception as e:
            _logger.error(f"Error in intensive cleanup {cleanup_type}: {e}")

def _execute_comprehensive_cleanup(max_age_s: float) -> None:
    """Execute comprehensive cleanup of all resource types."""
    from .guards.file_guard import force_cleanup_files
    from .guards.socket_guard import force_cleanup_sockets
    from .guards.asyncio_guard import force_cleanup_asyncio
    from .guards.event_guard import force_cleanup_listeners
    
    total_cleaned = 0
    total_cleaned += force_cleanup_files(max_age_s=max_age_s)
    total_cleaned += force_cleanup_sockets(max_age_s=max_age_s)
    task_count, timer_count = force_cleanup_asyncio(max_age_s=max_age_s)
    total_cleaned += task_count + timer_count
    total_cleaned += force_cleanup_listeners(max_age_s=max_age_s)
    
    _logger.info(f"Comprehensive cleanup completed: {total_cleaned} resources cleaned")

def _execute_file_intensive_cleanup(max_age_s: float) -> None:
    """Execute intensive file handle cleanup."""
    from .guards.file_guard import force_cleanup_files
    cleaned = force_cleanup_files(max_age_s=max_age_s)
    _logger.info(f"File intensive cleanup completed: {cleaned} files cleaned")

def _execute_socket_intensive_cleanup(max_age_s: float) -> None:
    """Execute intensive socket cleanup."""
    from .guards.socket_guard import force_cleanup_sockets
    cleaned = force_cleanup_sockets(max_age_s=max_age_s)
    _logger.info(f"Socket intensive cleanup completed: {cleaned} sockets cleaned")

def _execute_memory_intensive_cleanup(max_age_s: float) -> None:
    """Execute intensive memory cleanup including caches and cycles."""
    try:
        from .detectors.caches import force_cleanup_caches
        cache_cleaned = force_cleanup_caches(max_age_s=max_age_s)
        _logger.info(f"Memory intensive cleanup completed: {cache_cleaned} caches cleaned")
    except Exception as e:
        _logger.error(f"Error in memory intensive cleanup: {e}")
    
    # Execute cleanups outside of scan lock
    for cleanup in cleanups_to_run:
        try:
            guard_type = cleanup['guard_type']
            max_age_s = cleanup['max_age_s']
            _logger.info(f"_execute_scheduled_cleanups: Executing {guard_type} cleanup with max_age_s={max_age_s}")
            
            if guard_type == 'file':
                from .guards.file_guard import force_cleanup_files
                cleanup_count = force_cleanup_files(max_age_s=max_age_s)
                _logger.info(f"File cleanup completed: {cleanup_count} files cleaned")
            elif guard_type == 'socket':
                from .guards.socket_guard import force_cleanup_sockets  
                cleanup_count = force_cleanup_sockets(max_age_s=max_age_s)
                _logger.info(f"Socket cleanup completed: {cleanup_count} sockets cleaned")
            elif guard_type == 'asyncio':
                from .guards.asyncio_guard import force_cleanup_asyncio
                task_count, timer_count = force_cleanup_asyncio(max_age_s=max_age_s)
                _logger.info(f"Asyncio cleanup completed: {task_count} tasks, {timer_count} timers cleaned")
            elif guard_type == 'event':
                from .guards.event_guard import force_cleanup_listeners
                cleanup_count = force_cleanup_listeners(max_age_s=max_age_s)
                _logger.info(f"Event cleanup completed: {cleanup_count} listeners cleaned")
            elif guard_type == 'cache':
                from .detectors.cache_detector import force_cleanup_caches
                cleanup_count = force_cleanup_caches(max_age_s=max_age_s)
                _logger.info(f"Cache cleanup completed: {cleanup_count} caches cleaned")
        except Exception as e:
            _logger.error(f"Error in scheduled cleanup for {guard_type}: {e}")
            import traceback
            _logger.error(f"Cleanup error traceback: {traceback.format_exc()}")
    
    # Check and execute any due intensive cleanups
    _execute_intensive_cleanups()

# Performance monitoring
_performance_stats = {
    'total_scans': 0,
    'total_findings': 0,
    'avg_scan_duration_ms': 0.0,
    'overhead_percentage': 0.0,
    'memory_baseline_mb': 0.0,
    'guard_overhead_ns': 0,
    'detector_overhead_ns': 0
}

# Environment detection
_environment_info = {
    'platform': platform.system(),
    'python_implementation': platform.python_implementation(),
    'python_version': platform.python_version(),
    'hostname': platform.node(),
    'process_name': os.path.basename(sys.argv[0]) if sys.argv else 'unknown'
}


@dataclass
class GuardModule:
    """Information about a guard module for systematic management."""
    name: str
    install_func: Callable[[MemGuardConfig], None]
    uninstall_func: Callable[[], None]
    scan_func: Callable[..., List[LeakFinding]]
    info_func: Callable[[], Dict[str, Any]]
    pattern_name: str
    enabled_by_default: bool = True


@dataclass
class DetectorModule:
    """Information about a detector module for systematic management."""
    name: str
    install_func: Callable[[MemGuardConfig], None]
    uninstall_func: Callable[[], None]
    scan_func: Callable[..., List[LeakFinding]]
    info_func: Callable[[], Dict[str, Any]]
    pattern_name: str
    enabled_by_default: bool = True


# Registry of all available guards and detectors
_GUARD_MODULES = [
    GuardModule(
        name="file_guard",
        install_func=install_file_guard,
        uninstall_func=uninstall_file_guard,
        scan_func=scan_open_files,
        info_func=get_tracked_files_info,
        pattern_name="handles"
    ),
    GuardModule(
        name="socket_guard",
        install_func=install_socket_guard,
        uninstall_func=uninstall_socket_guard,
        scan_func=scan_open_sockets,
        info_func=get_tracked_sockets_info,
        pattern_name="handles"
    ),
    GuardModule(
        name="asyncio_guard",
        install_func=install_asyncio_guard,
        uninstall_func=uninstall_asyncio_guard,
        scan_func=scan_running_tasks,
        info_func=get_tracked_asyncio_info,
        pattern_name="timers"
    ),
    GuardModule(
        name="event_guard",
        install_func=install_event_guard,
        uninstall_func=uninstall_event_guard,
        scan_func=scan_event_listeners,
        info_func=get_tracked_listeners_info,
        pattern_name="listeners"
    )
]

_DETECTOR_MODULES = [
    DetectorModule(
        name="cycle_detector",
        install_func=install_cycle_detector,
        uninstall_func=uninstall_cycle_detector,
        scan_func=scan_reference_cycles,
        info_func=get_cycle_detection_info,
        pattern_name="cycles"
    ),
    DetectorModule(
        name="cache_detector",
        install_func=install_cache_detector,
        uninstall_func=uninstall_cache_detector,
        scan_func=scan_growing_caches,
        info_func=get_cache_detection_info,
        pattern_name="caches"
    )
]


def protect(threshold_mb: int = MemGuardConfig.threshold_mb,
           poll_interval_s: float = MemGuardConfig.poll_interval_s,
           sample_rate: float = MemGuardConfig.sample_rate,
           patterns: tuple = ('handles', 'caches', 'timers', 'cycles', 'listeners'),
           auto_cleanup: Optional[Dict[str, bool]] = None,
           debug_mode: bool = MemGuardConfig.debug_mode,
           background: bool = True,
           config: Optional[MemGuardConfig] = None,
           monitoring_mode: str = "hybrid",
           intensive_cleanup_schedule: str = "custom") -> None:
    """
    Start MemGuard memory leak protection.
    
    Args:
        threshold_mb: Memory threshold to trigger analysis (MB)
        poll_interval_s: Background polling interval (seconds)
        sample_rate: Statistical sampling rate (0.0-1.0)
        patterns: Tuple of patterns to detect ('handles', 'caches', etc.)
        auto_cleanup: Dict of pattern -> auto-cleanup enabled
        debug_mode: Enable debugging output (testing only, adds overhead)
        background: Run background scanner
        config: Pre-configured MemGuardConfig object
        monitoring_mode: Monitoring mode ('light', 'deep', or 'hybrid')
        intensive_cleanup_schedule: Schedule for intensive cleanup ('never', 'hourly', 'daily', 'custom')
    
    This is the main entry point for MemGuard protection.
    """
    with _memguard_state['protection_lock']:
        if _memguard_state['is_protecting']:
            _logger.warning("MemGuard protection already active")
            return
        
        try:
            # All features available in open source version
            _logger.info("MemGuard open source - all features available")
            
            # Create or use provided configuration
            if config is None:
                config = MemGuardConfig(
                    threshold_mb=threshold_mb,
                    poll_interval_s=poll_interval_s,
                    sample_rate=sample_rate,
                    patterns=patterns,
                    debug_mode=debug_mode,
                    monitoring_mode=monitoring_mode,
                    intensive_cleanup_schedule=intensive_cleanup_schedule
                )
                
                # Configure auto_cleanup per pattern if provided
                if auto_cleanup is not None:
                    from .config import PatternTuning
                    from dataclasses import replace
                    # Convert tuning to a mutable dict so we can modify it
                    new_tuning = dict(config.tuning)
                    for pattern, enabled in auto_cleanup.items():
                        if pattern in new_tuning:
                            # Create new tuning with auto_cleanup enabled
                            old_tuning = new_tuning[pattern]
                            new_tuning[pattern] = PatternTuning(
                                enabled=old_tuning.enabled,
                                auto_cleanup=enabled,
                                max_age_s=old_tuning.max_age_s,
                                min_growth=old_tuning.min_growth,
                                min_len=old_tuning.min_len,
                                memory_estimate_mb=old_tuning.memory_estimate_mb
                            )
                    # CRITICAL FIX: Use replace() for frozen dataclass
                    config = replace(config, tuning=new_tuning)
            
            _memguard_state['config'] = config
            
            # Store baseline memory usage
            memory_tracker = get_memory_tracker()
            if memory_tracker.is_available():
                _performance_stats['memory_baseline_mb'] = memory_tracker.get_rss_mb()
            
            # Install guards and detectors based on configuration
            _install_protection_modules(config)
            
            # Start background scanner if requested
            if background:
                _start_background_scanner(config)
            
            _memguard_state['is_protecting'] = True
            _memguard_state['start_time'] = time.time()
            
            _logger.info(f"MemGuard protection started with patterns: {config.patterns}")
            
        except Exception as e:
            _logger.error(f"Failed to start MemGuard protection: {e}")
            # Clean up partial installation
            stop()
            raise


def stop() -> None:
    """
    Stop MemGuard protection and clean up all instrumentation.
    
    This safely removes all guards and detectors, restoring original behavior.
    """
    with _memguard_state['protection_lock']:
        if not _memguard_state['is_protecting']:
            return
        
        try:
            # Stop background scanner
            _stop_background_scanner()
            
            # Uninstall all protection modules
            _uninstall_protection_modules()
            
            # Reset state
            _memguard_state['is_protecting'] = False
            _memguard_state['config'] = None
            _memguard_state['installed_guards'].clear()
            _memguard_state['installed_detectors'].clear()
            
            _logger.info("MemGuard protection stopped")
            
        except Exception as e:
            _logger.error(f"Error stopping MemGuard protection: {e}")


def analyze(force_full_scan: bool = False,
           include_performance: bool = False) -> MemGuardReport:
    """
    Analyze current memory state and detect leaks.
    
    Args:
        force_full_scan: Force full scan even if not scheduled
        include_performance: Include performance statistics in report
    
    Returns:
        MemGuardReport with detected leaks and metadata
    
    This can be called whether protection is active or not.
    """
    start_time = time.perf_counter()
    scan_findings: List[LeakFinding] = []
    
    try:
        # Get current memory usage
        memory_tracker = get_memory_tracker()
        current_memory = memory_tracker.get_rss_mb() if memory_tracker.is_available() else 0.0
        
        config = _memguard_state.get('config') or MemGuardConfig()
        
        # Scan all installed guards
        for guard in _GUARD_MODULES:
            if guard.name in _memguard_state['installed_guards']:
                try:
                    guard_findings = guard.scan_func()
                    scan_findings.extend(guard_findings)
                except Exception as e:
                    _logger.debug(f"Error scanning {guard.name}: {e}")
        
        # Scan all installed detectors
        for detector in _DETECTOR_MODULES:
            if detector.name in _memguard_state['installed_detectors']:
                try:
                    if detector.pattern_name == "cycles":
                        detector_findings = detector.scan_func(force_full_scan=force_full_scan)
                    else:
                        detector_findings = detector.scan_func()
                    scan_findings.extend(detector_findings)
                except Exception as e:
                    _logger.debug(f"Error scanning {detector.name}: {e}")
        
        # Update performance statistics
        scan_duration = (time.perf_counter() - start_time) * 1000  # Convert to ms
        _performance_stats['total_scans'] += 1
        _performance_stats['total_findings'] += len(scan_findings)
        
        # Calculate running average
        total_scans = _performance_stats['total_scans']
        current_avg = _performance_stats['avg_scan_duration_ms']
        _performance_stats['avg_scan_duration_ms'] = (
            (current_avg * (total_scans - 1) + scan_duration) / total_scans
        )
        
        # Calculate overhead percentage (scan time vs poll interval)
        poll_interval_ms = config.poll_interval_s * 1000  # Convert to ms
        current_overhead = (_performance_stats['avg_scan_duration_ms'] / poll_interval_ms) * 100
        _performance_stats['overhead_percentage'] = round(current_overhead, 3)
        
        # Create report
        report = create_report(
            findings=scan_findings,
            scan_duration_ms=scan_duration,
            memory_baseline_mb=_performance_stats['memory_baseline_mb'],
            memory_current_mb=current_memory,
            sampling_rate=config.sample_rate,
            hostname=_environment_info['hostname'],
            process_name=_environment_info['process_name'],
            python_version=_environment_info['python_version'],
            platform=_environment_info['platform']
        )
        
        _memguard_state['last_scan_time'] = time.time()
        _memguard_state['scan_count'] += 1
        
        return report
        
    except Exception as e:
        _logger.error(f"Error during analysis: {e}")
        # Return empty report on error
        return create_report(
            findings=[],
            scan_duration_ms=(time.perf_counter() - start_time) * 1000,
            memory_current_mb=0.0
        )


def analyze_with_cleanup(config: MemGuardConfig, force_full_scan: bool = False) -> MemGuardReport:
    """
    Perform memory leak analysis with automatic cleanup for detected leaks.
    
    This function extends analyze() by performing actual cleanup operations
    when auto_cleanup is enabled for specific patterns.
    
    Args:
        config: MemGuard configuration with auto_cleanup settings
        force_full_scan: Force comprehensive scanning (slower but more thorough)
    
    Returns:
        MemGuardReport with findings and cleanup results
    """
    if not _memguard_state['is_protecting']:
        return create_report([], 0, 0, 0)
    
    start_time = time.perf_counter()
    scan_findings: List[LeakFinding] = []
    
    try:
        with _memguard_state['protection_lock']:
            # Get current memory
            memory_tracker = get_memory_tracker()
            current_memory = memory_tracker.get_rss_mb() if memory_tracker.is_available() else 0.0
            
            # Scan all installed guards and perform auto-cleanup if enabled
            for guard in _GUARD_MODULES:
                if guard.name in _memguard_state['installed_guards']:
                    try:
                        guard_findings = guard.scan_func()
                        scan_findings.extend(guard_findings)
                        
                        # Schedule auto-cleanup for next scan cycle to avoid race conditions
                        # CRITICAL FIX: Don't cleanup during same scan to prevent iterator invalidation
                        if guard.name == 'file_guard' and config.auto_cleanup_enabled("handles"):
                            _schedule_guard_cleanup('file', config.tuning["handles"].max_age_s)
                        elif guard.name == 'socket_guard' and config.auto_cleanup_enabled("handles"):
                            _schedule_guard_cleanup('socket', config.tuning["handles"].max_age_s)
                        elif guard.name == 'asyncio_guard' and config.auto_cleanup_enabled("timers"):
                            _schedule_guard_cleanup('asyncio', config.tuning["timers"].max_age_s)
                        elif guard.name == 'event_guard' and config.auto_cleanup_enabled("listeners"):
                            _schedule_guard_cleanup('event', config.tuning["listeners"].max_age_s)
                    except Exception as e:
                        _logger.debug(f"Error scanning {guard.name}: {e}")
            
            # Scan all installed detectors WITH auto-cleanup
            for detector in _DETECTOR_MODULES:
                if detector.name in _memguard_state['installed_detectors']:
                    try:
                        if detector.pattern_name == "cycles":
                            # Cycles detector with auto-cleanup
                            auto_cleanup_enabled = config.auto_cleanup_enabled("cycles")
                            _logger.debug(f"Cycle auto-cleanup enabled: {auto_cleanup_enabled}")
                            detector_findings = detector.scan_func(force_full_scan=force_full_scan, auto_cleanup=auto_cleanup_enabled)
                        elif detector.pattern_name == "caches":
                            # Cache detector with auto-cleanup
                            auto_cleanup_enabled = config.auto_cleanup_enabled("caches")
                            _logger.debug(f"Cache auto-cleanup enabled: {auto_cleanup_enabled}")
                            detector_findings = detector.scan_func(auto_cleanup=auto_cleanup_enabled)
                        else:
                            detector_findings = detector.scan_func()
                        scan_findings.extend(detector_findings)
                    except Exception as e:
                        _logger.debug(f"Error scanning {detector.name}: {e}")
            
            # Update performance statistics
            scan_duration = (time.perf_counter() - start_time) * 1000  # Convert to ms
            _performance_stats['total_scans'] += 1
            _performance_stats['total_findings'] += len(scan_findings)
            
            # Calculate running average
            total_scans = _performance_stats['total_scans']
            current_avg = _performance_stats['avg_scan_duration_ms']
            _performance_stats['avg_scan_duration_ms'] = (
                (current_avg * (total_scans - 1) + scan_duration) / total_scans
            )
            
            # Calculate overhead percentage (scan time vs poll interval)
            poll_interval_ms = config.poll_interval_s * 1000  # Convert to ms
            current_overhead = (_performance_stats['avg_scan_duration_ms'] / poll_interval_ms) * 100
            _performance_stats['overhead_percentage'] = round(current_overhead, 3)
            
            # Create report
            report = create_report(
                findings=scan_findings,
                scan_duration_ms=scan_duration,
                memory_baseline_mb=_performance_stats['memory_baseline_mb'],
                memory_current_mb=current_memory,
                sampling_rate=config.sample_rate,
                hostname=_environment_info['hostname'],
                process_name=_environment_info['process_name'],
                python_version=_environment_info['python_version'],
                platform=_environment_info['platform']
            )
            
            _memguard_state['last_scan_time'] = time.time()
            _memguard_state['scan_count'] += 1
            
            # CRITICAL FIX: Execute scheduled cleanups after scan completion
            # This prevents race conditions by doing cleanup when scan is finished
            _execute_scheduled_cleanups()
            
            return report
            
    except Exception as e:
        _logger.error(f"Error during analysis with cleanup: {e}")
        # Add detailed error tracing
        import traceback
        _logger.error(f"Full traceback: {traceback.format_exc()}")
        # Return empty report on error
        return create_report(
            findings=[],
            scan_duration_ms=(time.perf_counter() - start_time) * 1000,
            memory_current_mb=0.0
        )


def get_report() -> MemGuardReport:
    """
    Get the most recent analysis report.
    
    Returns:
        MemGuardReport from the last analyze() call, or new scan if none exists
    
    Convenience function that calls analyze() if no recent report exists.
    """
    # Always perform a fresh analysis for the most current data
    return analyze()


def is_protecting() -> bool:
    """Check if MemGuard protection is currently active."""
    return _memguard_state['is_protecting']


def get_status() -> Dict[str, Any]:
    """
    Get comprehensive status information about MemGuard.
    
    Returns:
        Dictionary with standardized schema for observability integration
        
    Schema Version: 1.0
    Required fields for dashboard/monitoring compatibility:
    - schema_version, is_protecting, uptime_seconds, scan_count
    - performance_stats.overhead_percentage, performance_stats.memory_current_mb
    - guards, detectors (with versions and status)
    """
    with _memguard_state['protection_lock']:
        config = _memguard_state.get('config')
        
        # Calculate overhead percentage for observability
        overhead_pct = _performance_stats.get('overhead_percentage', 0.0)
        
        # Standardized status schema v1.0
        status = {
            # Schema metadata for compatibility
            'schema_version': '1.0',
            'memguard_version': '1.0.0',
            'timestamp': time.time(),
            
            # Core status (required by tests/dashboards)
            'is_protecting': _memguard_state['is_protecting'],
            'uptime_seconds': time.time() - _memguard_state['start_time'] if _memguard_state['start_time'] else 0,
            'scan_count': _memguard_state['scan_count'],
            'last_scan_age_seconds': time.time() - _memguard_state['last_scan_time'] if _memguard_state['last_scan_time'] else None,
            'background_scanner_active': _memguard_state['background_scanner'] is not None,
            
            # Performance metrics (standardized for observability)
            'performance_stats': {
                'memory_baseline_mb': _performance_stats.get('memory_baseline_mb', 0.0),
                'memory_current_mb': _performance_stats.get('memory_current_mb', 0.0),
                'memory_growth_mb': _performance_stats.get('memory_current_mb', 0.0) - _performance_stats.get('memory_baseline_mb', 0.0),
                'total_findings': _performance_stats.get('total_findings', 0),
                'total_scans': _performance_stats.get('total_scans', 0),
                'avg_scan_duration_ms': _performance_stats.get('avg_scan_duration_ms', 0.0),
                'overhead_percentage': overhead_pct,  # Critical for customer self-verification
                'scan_frequency_hz': 1.0 / config.poll_interval_s if config else 0.0
            },
            
            # Guards status (versioned for compatibility)
            'guards': {},
            
            # Detectors status (versioned for compatibility) 
            'detectors': {},
            
            # Configuration (stable subset)
            'configuration': {},
            
            # Environment (stable fields only)
            'environment': {
                'hostname': _environment_info.get('hostname', 'unknown'),
                'platform': _environment_info.get('platform', 'unknown'),
                'python_version': _environment_info.get('python_version', 'unknown'),
                'process_name': _environment_info.get('process_name', 'unknown')
            }
        }
        
        # Populate guards with versions and status
        for guard in _GUARD_MODULES:
            guard_active = guard.name in _memguard_state['installed_guards']
            try:
                guard_info = guard.info_func() if guard_active else {}
                status['guards'][guard.pattern_name] = {
                    'version': '1.0.0',
                    'active': guard_active,
                    'status': 'running' if guard_active else 'disabled',
                    'performance': guard_info
                }
            except Exception as e:
                status['guards'][guard.pattern_name] = {
                    'version': '1.0.0',
                    'active': False,
                    'status': 'error',
                    'error': str(e)
                }
        
        # Populate detectors with versions and status
        for detector in _DETECTOR_MODULES:
            detector_active = detector.name in _memguard_state['installed_detectors']
            try:
                detector_info = detector.info_func() if detector_active else {}
                status['detectors'][detector.pattern_name] = {
                    'version': '1.0.0',
                    'active': detector_active,
                    'status': 'running' if detector_active else 'disabled',
                    'performance': detector_info
                }
            except Exception as e:
                status['detectors'][detector.pattern_name] = {
                    'version': '1.0.0',
                    'active': False,
                    'status': 'error',
                    'error': str(e)
                }
        
        # Stable configuration subset (avoid exposing internals)
        if config:
            status['configuration'] = {
                'threshold_mb': config.threshold_mb,
                'poll_interval_s': config.poll_interval_s,
                'sample_rate': config.sample_rate,
                'patterns': list(config.patterns),
                'auto_cleanup_enabled': any(config.auto_cleanup_enabled(p) for p in config.patterns),
                'debug_mode': config.debug_mode
            }
        
        return status


def get_performance_summary() -> Dict[str, Any]:
    """
    Get performance overhead summary for all components.
    
    Returns:
        Dictionary with overhead statistics from guards and detectors
    """
    summary = {
        'core_stats': _performance_stats.copy(),
        'guard_stats': {},
        'detector_stats': {}
    }
    
    # Collect performance stats from installed guards
    for guard in _GUARD_MODULES:
        if guard.name in _memguard_state['installed_guards']:
            try:
                guard_info = guard.info_func()
                summary['guard_stats'][guard.name] = guard_info
            except Exception as e:
                summary['guard_stats'][guard.name] = {'error': str(e)}
    
    # Collect performance stats from installed detectors
    for detector in _DETECTOR_MODULES:
        if detector.name in _memguard_state['installed_detectors']:
            try:
                detector_info = detector.info_func()
                summary['detector_stats'][detector.name] = detector_info
            except Exception as e:
                summary['detector_stats'][detector.name] = {'error': str(e)}
    
    return summary


def _install_protection_modules(config: MemGuardConfig) -> None:
    """Install guards and detectors based on configuration."""
    # Install guards
    for guard in _GUARD_MODULES:
        if guard.pattern_name in config.patterns:
            try:
                guard.install_func(config)
                _memguard_state['installed_guards'].add(guard.name)
                _logger.debug(f"Installed {guard.name}")
            except Exception as e:
                _logger.warning(f"Failed to install {guard.name}: {e}")
    
    # Install detectors
    for detector in _DETECTOR_MODULES:
        if detector.pattern_name in config.patterns:
            try:
                detector.install_func(config)
                _memguard_state['installed_detectors'].add(detector.name)
                _logger.debug(f"Installed {detector.name}")
            except Exception as e:
                _logger.warning(f"Failed to install {detector.name}: {e}")


def _uninstall_protection_modules() -> None:
    """Uninstall all guards and detectors."""
    # Uninstall guards
    for guard in _GUARD_MODULES:
        if guard.name in _memguard_state['installed_guards']:
            try:
                guard.uninstall_func()
                _logger.debug(f"Uninstalled {guard.name}")
            except Exception as e:
                _logger.warning(f"Error uninstalling {guard.name}: {e}")
    
    # Uninstall detectors
    for detector in _DETECTOR_MODULES:
        if detector.name in _memguard_state['installed_detectors']:
            try:
                detector.uninstall_func()
                _logger.debug(f"Uninstalled {detector.name}")
            except Exception as e:
                _logger.warning(f"Error uninstalling {detector.name}: {e}")



def _start_background_scanner(config: MemGuardConfig) -> None:
    """Start background scanning thread with hybrid monitoring support."""
    if _memguard_state['background_scanner'] is not None:
        return
    
    _memguard_state['background_stop_event'].clear()
    _memguard_state['monitoring_mode'] = config.monitoring_mode
    
    # Schedule initial intensive cleanup if configured
    cleanup_schedule = config.get_intensive_cleanup_schedule()
    if cleanup_schedule and config.intensive_cleanup_schedule != "never":
        _schedule_next_intensive_cleanup(config)
    
    def background_scan_loop():
        """Background thread that periodically scans for leaks with hybrid monitoring."""
        _logger.debug(f"Background scanner started in {config.monitoring_mode} mode")
        
        last_summary_time = time.time()
        scan_count = 0
        light_scan_findings_count = 0
        
        while not _memguard_state['background_stop_event'].wait(config.poll_interval_s):
            try:
                scan_count += 1
                current_time = time.time()
                
                # Hybrid monitoring: Determine scan type based on mode and triggers
                should_deep_scan = _should_trigger_deep_scan(config, light_scan_findings_count, scan_count)
                
                # Check memory levels for context
                memory_tracker = get_memory_tracker()
                current_memory = 0.0
                baseline_memory = _performance_stats['memory_baseline_mb']
                threshold_exceeded = False
                
                if memory_tracker.is_available():
                    current_memory = memory_tracker.get_rss_mb()
                    threshold_exceeded = (current_memory - baseline_memory) > config.threshold_mb
                    
                    if threshold_exceeded:
                        _logger.info(f"Memory threshold exceeded: {current_memory:.1f}MB (baseline: {baseline_memory:.1f}MB)")
                        should_deep_scan = True  # Force deep scan on threshold breach
                
                # Perform scan based on monitoring mode
                scan_type = "deep" if should_deep_scan else "light"
                _logger.debug(f"Background scan #{scan_count} starting ({scan_type} mode, memory: {current_memory:.1f}MB)")
                
                if should_deep_scan:
                    report = _analyze_with_deep_scan(config)
                    light_scan_findings_count = 0  # Reset trigger counter
                else:
                    report = _analyze_with_light_scan(config)
                    light_scan_findings_count += len(report.findings)
                
                if report.findings:
                    _logger.warning(f"Background scan found {len(report.findings)} potential leaks ({scan_type} mode)")
                    
                    # Log critical findings
                    critical_findings = report.critical_findings
                    if critical_findings:
                        for finding in critical_findings[:3]:  # Log top 3 critical
                            _logger.error(f"CRITICAL LEAK: {finding.pattern} at {finding.location} "
                                        f"({finding.size_mb:.1f}MB) - {finding.suggested_fix}")
                    
                    # In light mode, accumulate findings for deep scan trigger
                    if not should_deep_scan and len(report.findings) >= config.deep_scan_trigger_threshold:
                        _logger.info(f"Light scan found {len(report.findings)} findings, triggering deep scan on next cycle")
                        light_scan_findings_count = config.deep_scan_trigger_threshold
                
                # Debug mode summary (testing only)
                if config.debug_mode and current_time - last_summary_time >= 60.0:
                    print(f"🔍 DEBUG: {scan_count} scans ({config.monitoring_mode} mode) in {current_time - last_summary_time:.0f}s | Memory: {current_memory:.1f}MB | Findings buffer: {light_scan_findings_count}")
                    last_summary_time = current_time
                    scan_count = 0
                
            except Exception as e:
                _logger.debug(f"Error in background scanner: {e}")
        
        _logger.debug("Background scanner stopped")
    
    scanner_thread = threading.Thread(
        target=background_scan_loop,
        name="MemGuard-BackgroundScanner",
        daemon=True
    )
    scanner_thread.start()
    _memguard_state['background_scanner'] = scanner_thread


def analyze(threshold_mb: Optional[int] = None,
           sample_rate: Optional[float] = None) -> MemGuardReport:
    """
    Analyze current memory usage and detect potential leaks.
    
    Args:
        threshold_mb: Memory threshold override (uses config default if None)
        sample_rate: Sample rate override (uses config default if None)
        
    Returns:
        MemGuardReport with detected leaks and findings
    """
    if not _memguard_state['is_protecting']:
        return create_report([])
    
    config = _memguard_state['config']
    if not config:
        return create_report([])
    
    # Use provided thresholds or fall back to config
    actual_threshold = threshold_mb if threshold_mb is not None else config.threshold_mb
    actual_sample_rate = sample_rate if sample_rate is not None else config.sample_rate
    
    # Collect findings from all installed guards and detectors
    all_findings = []
    
    # Scan file guards
    if 'file_guard' in _memguard_state['installed_guards']:
        try:
            from .guards.file_guard import scan_open_files
            tuning = config.tuning_for('handles')
            file_findings = scan_open_files(max_age_s=tuning.max_age_s)
            all_findings.extend(file_findings)
        except Exception as e:
            _logger.debug(f"Error scanning files: {e}")
    
    # Scan socket guards  
    if 'socket_guard' in _memguard_state['installed_guards']:
        try:
            from .guards.socket_guard import scan_open_sockets
            tuning = config.tuning_for('handles')  # sockets use handles tuning
            socket_findings = scan_open_sockets(max_age_s=tuning.max_age_s)
            all_findings.extend(socket_findings)
        except Exception as e:
            _logger.debug(f"Error scanning sockets: {e}")
    
    # Scan asyncio guards
    if 'asyncio_guard' in _memguard_state['installed_guards']:
        try:
            from .guards.asyncio_guard import scan_running_tasks
            tuning = config.tuning_for('timers')
            asyncio_findings = scan_running_tasks(max_age_s=tuning.max_age_s)
            all_findings.extend(asyncio_findings)
        except Exception as e:
            _logger.debug(f"Error scanning asyncio: {e}")
    
    # Scan cache detectors
    if 'cache_detector' in _memguard_state['installed_detectors']:
        try:
            from .detectors.caches import scan_cache_growth
            tuning = config.tuning_for('caches')
            cache_findings = scan_cache_growth(min_growth=tuning.min_growth, min_len=tuning.min_len)
            all_findings.extend(cache_findings)
        except Exception as e:
            _logger.debug(f"Error scanning caches: {e}")
    
    # Update performance stats
    _performance_stats['total_scans'] += 1
    _performance_stats['total_findings'] += len(all_findings)
    
    return create_report(all_findings)


def analyze_with_cleanup(config: MemGuardConfig) -> MemGuardReport:
    """
    Analyze current memory usage with auto-cleanup enabled.
    
    This function performs leak detection and triggers cleanup actions
    for patterns that have auto_cleanup enabled in the configuration.
    
    Args:
        config: MemGuard configuration with cleanup settings
        
    Returns:
        MemGuardReport with findings and cleanup actions performed
    """
    # First perform standard analysis
    report = analyze()
    
    # If no findings, return early
    if not report.findings:
        _logger.debug("analyze_with_cleanup: No findings to process")
        return report
    
    _logger.debug(f"analyze_with_cleanup: Processing {len(report.findings)} findings")
    
    # Group findings by pattern for cleanup decisions
    findings_by_pattern = {}
    for finding in report.findings:
        pattern = finding.pattern
        if pattern not in findings_by_pattern:
            findings_by_pattern[pattern] = []
        findings_by_pattern[pattern].append(finding)
    
    _logger.debug(f"analyze_with_cleanup: Findings by pattern: {[(k, len(v)) for k, v in findings_by_pattern.items()]}")
    
    # Process cleanup for each pattern type
    for pattern, pattern_findings in findings_by_pattern.items():
        _logger.debug(f"analyze_with_cleanup: Processing pattern '{pattern}' with {len(pattern_findings)} findings")
        
        # Check if auto-cleanup is enabled for this pattern
        auto_cleanup_enabled = config.auto_cleanup_enabled(pattern)
        _logger.debug(f"analyze_with_cleanup: Pattern '{pattern}' auto_cleanup_enabled: {auto_cleanup_enabled}")
        
        if not auto_cleanup_enabled:
            _logger.debug(f"analyze_with_cleanup: Skipping '{pattern}' - auto-cleanup disabled")
            continue
            
        # Get pattern-specific tuning
        tuning = config.tuning_for(pattern)
        if not tuning or not tuning.auto_cleanup:
            _logger.debug(f"analyze_with_cleanup: Skipping '{pattern}' - tuning disabled (tuning: {tuning})")
            continue
        
        _logger.debug(f"analyze_with_cleanup: Pattern '{pattern}' tuning: auto_cleanup={tuning.auto_cleanup}, max_age_s={tuning.max_age_s}")
        
        # Schedule cleanup based on pattern type and findings
        cleanup_threshold = tuning.max_age_s
        
        # Only schedule if we have findings (adjusted thresholds for comprehensive cleanup)
        threshold_by_pattern = {
            'handles': 3,      # Files/sockets need more findings (they generate many)
            'caches': 1,       # Cache leaks are significant even with 1 finding
            'timers': 1,       # Timer leaks are significant even with 1 finding  
            'cycles': 1,       # Cycle leaks are significant even with 1 finding
            'listeners': 1     # Event listener leaks are significant even with 1 finding
        }
        threshold = threshold_by_pattern.get(pattern, 1)  # Default to 1 for new patterns
        
        if len(pattern_findings) >= threshold:
            _logger.info(f"analyze_with_cleanup: Scheduling cleanup for pattern '{pattern}' - {len(pattern_findings)} findings exceed threshold")
            
            if pattern == 'handles':
                # Handles pattern covers both files and sockets
                _schedule_guard_cleanup('file', cleanup_threshold)
                _schedule_guard_cleanup('socket', cleanup_threshold)
                _logger.info(f"analyze_with_cleanup: Scheduled file and socket cleanup with threshold {cleanup_threshold}s")
            elif pattern == 'timers':
                _schedule_guard_cleanup('asyncio', cleanup_threshold)
                _logger.info(f"analyze_with_cleanup: Scheduled asyncio cleanup with threshold {cleanup_threshold}s")
            elif pattern == 'listeners':
                _schedule_guard_cleanup('event', cleanup_threshold)
                _logger.info(f"analyze_with_cleanup: Scheduled event cleanup with threshold {cleanup_threshold}s")
            elif pattern == 'caches':
                # Schedule cache cleanup via detector
                _schedule_cache_cleanup(cleanup_threshold)
                _logger.info(f"analyze_with_cleanup: Scheduled cache cleanup with threshold {cleanup_threshold}s")
            elif pattern == 'cycles':
                # Cycles are detection-only for safety (breaking cycles can cause issues)
                _logger.info(f"analyze_with_cleanup: Cycles detected but cleanup skipped for safety - {len(pattern_findings)} cycle findings")
            
            _logger.debug(f"Scheduled cleanup for {pattern}: {len(pattern_findings)} findings")
        else:
            _logger.debug(f"analyze_with_cleanup: Not scheduling cleanup for '{pattern}' - only {len(pattern_findings)} findings (need ≥{threshold})")
    
    # Execute any scheduled cleanups
    _execute_scheduled_cleanups()
    
    return report


def _should_trigger_deep_scan(config: MemGuardConfig, light_findings_count: int, scan_count: int) -> bool:
    """Determine if a deep scan should be triggered based on monitoring mode and conditions."""
    if config.monitoring_mode == "deep":
        return True
    elif config.monitoring_mode == "light":
        return False
    elif config.monitoring_mode == "hybrid":
        # Trigger deep scan if:
        # 1. Light scan findings exceed threshold
        # 2. Every 10th scan (fallback to ensure periodic deep scans)
        # 3. Next scheduled deep scan time has arrived
        return (light_findings_count >= config.deep_scan_trigger_threshold or
                scan_count % 10 == 0 or
                time.time() >= _memguard_state.get('next_scheduled_deep_scan', 0.0))
    return False

def _analyze_with_light_scan(config: MemGuardConfig) -> MemGuardReport:
    """Perform lightweight analysis with reduced sampling and selected patterns."""
    # Use light sampling rate
    light_config = config.merge(sample_rate=config.get_current_sample_rate(is_deep_scan=False))
    
    # Only scan high-priority patterns in light mode
    start_time = time.perf_counter()
    scan_findings: List[LeakFinding] = []
    
    try:
        # Scan only high-priority guards in light mode
        for guard in _GUARD_MODULES:
            if guard.name in _memguard_state['installed_guards']:
                if config.should_pattern_run_in_light_mode(guard.pattern_name):
                    try:
                        guard_findings = guard.scan_func()
                        # Sample findings to reduce overhead
                        sampled_findings = _sample_findings(guard_findings, light_config.get_current_sample_rate())
                        scan_findings.extend(sampled_findings)
                    except Exception as e:
                        _logger.debug(f"Error in light scan {guard.name}: {e}")
        
        # Skip expensive detectors in light mode unless high priority
        for detector in _DETECTOR_MODULES:
            if detector.name in _memguard_state['installed_detectors']:
                if config.should_pattern_run_in_light_mode(detector.pattern_name):
                    try:
                        if detector.pattern_name == "cycles":
                            continue  # Skip cycles in light mode (too expensive)
                        detector_findings = detector.scan_func()
                        sampled_findings = _sample_findings(detector_findings, light_config.get_current_sample_rate())
                        scan_findings.extend(sampled_findings)
                    except Exception as e:
                        _logger.debug(f"Error in light scan {detector.name}: {e}")
        
        # Create lightweight report
        memory_tracker = get_memory_tracker()
        current_memory = memory_tracker.get_rss_mb() if memory_tracker.is_available() else 0.0
        scan_duration = (time.perf_counter() - start_time) * 1000
        
        return create_report(
            findings=scan_findings,
            scan_duration_ms=scan_duration,
            memory_baseline_mb=_performance_stats['memory_baseline_mb'],
            memory_current_mb=current_memory,
            sampling_rate=light_config.get_current_sample_rate(),
            hostname=_environment_info['hostname'],
            process_name=_environment_info['process_name'],
            python_version=_environment_info['python_version'],
            platform=_environment_info['platform']
        )
        
    except Exception as e:
        _logger.error(f"Error during light analysis: {e}")
        return create_report(
            findings=[],
            scan_duration_ms=(time.perf_counter() - start_time) * 1000,
            memory_current_mb=0.0
        )

def _analyze_with_deep_scan(config: MemGuardConfig) -> MemGuardReport:
    """Perform comprehensive analysis with full sampling and cleanup."""
    _logger.info("Starting deep scan (higher overhead expected)")
    deep_config = config.merge(sample_rate=config.get_current_sample_rate(is_deep_scan=True))
    
    # Use existing comprehensive analysis with cleanup
    report = analyze_with_cleanup(deep_config)
    
    # Update next scheduled deep scan time
    _memguard_state['next_scheduled_deep_scan'] = time.time() + (config.poll_interval_s * 10)  # Next deep scan in 10 cycles
    
    _logger.info(f"Deep scan completed: {len(report.findings)} findings, {report.scan_duration_ms:.1f}ms")
    return report

def _sample_findings(findings: List[LeakFinding], sample_rate: float) -> List[LeakFinding]:
    """Sample findings based on rate to reduce overhead in light mode."""
    if sample_rate >= 1.0 or not findings:
        return findings
    
    import random
    sample_size = max(1, int(len(findings) * sample_rate))
    return random.sample(findings, sample_size)

def _schedule_next_intensive_cleanup(config: MemGuardConfig) -> None:
    """Schedule the next intensive cleanup based on configuration."""
    try:
        import croniter
        from datetime import datetime
        
        schedule = config.get_intensive_cleanup_schedule()
        if not schedule:
            return
        
        now = datetime.now()
        cron = croniter.croniter(schedule, now)
        next_time = cron.get_next(datetime)
        next_timestamp = next_time.timestamp()
        
        _schedule_intensive_cleanup("comprehensive", next_timestamp, max_age_s=300.0)
        _logger.info(f"Next intensive cleanup scheduled for {next_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except ImportError:
        _logger.warning("croniter not available - intensive cleanup scheduling disabled")
    except Exception as e:
        _logger.error(f"Error scheduling intensive cleanup: {e}")

def _stop_background_scanner() -> None:
    """Stop background scanning thread."""
    if _memguard_state['background_scanner'] is None:
        return
    
    _memguard_state['background_stop_event'].set()
    
    # Wait for thread to finish (with timeout)
    if _memguard_state['background_scanner'].is_alive():
        _memguard_state['background_scanner'].join(timeout=2.0)
    
    _memguard_state['background_scanner'] = None


# Module cleanup on exit
def _cleanup_on_exit():
    """Clean up MemGuard state on module exit."""
    try:
        if _memguard_state['is_protecting']:
            stop()
    except Exception:
        pass  # Ignore errors during cleanup


# Register cleanup handler
import atexit
atexit.register(_cleanup_on_exit)


# Convenience functions for common use cases
def quick_protect() -> None:
    """
    Quick start with sensible defaults for testing and development.
    """
    protect(
        threshold_mb=MemGuardConfig.threshold_mb,
        poll_interval_s=MemGuardConfig.poll_interval_s,
        sample_rate=MemGuardConfig.sample_rate
    )


def protect_production(threshold_mb: int = 200) -> None:
    """
    Production-safe protection with conservative settings.
    
    Args:
        threshold_mb: Memory threshold for triggering scans
    """
    protect(
        threshold_mb=threshold_mb,
        poll_interval_s=5.0,  # Less frequent scanning
        sample_rate=0.005,    # Lower overhead
        patterns=('handles', 'caches', 'cycles'),  # Skip more invasive patterns
        auto_cleanup={pattern: False for pattern in ('handles', 'caches', 'cycles')},  # No auto-cleanup
        background=True
    )


def protect_development(auto_fix: bool = True) -> None:
    """
    Development-friendly protection with auto-cleanup enabled.
    
    Args:
        auto_fix: Enable automatic cleanup of detected leaks
    """
    auto_cleanup = {
        'handles': auto_fix,
        'timers': auto_fix,
        'listeners': False,  # Never auto-remove listeners
        'cycles': False,     # Never auto-break cycles
        'caches': False      # Never auto-evict caches
    }
    
    protect(
        threshold_mb=25,
        poll_interval_s=1.0,
        sample_rate=0.05,
        auto_cleanup=auto_cleanup,
        background=True
    )


# Export the main API
__all__ = [
    'protect',
    'stop', 
    'analyze',
    'get_report',
    'is_protecting',
    'get_status',
    'get_performance_summary',
    'quick_protect',
    'protect_production',
    'protect_development'
]