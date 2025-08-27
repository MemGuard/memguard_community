#=============================================================================
# File        : memguard/detectors/__init__.py
# Project     : MemGuard v1.0
# Component   : Detectors Package - Memory Leak Pattern Detection Exports
# Description : Package initialization for memory leak pattern detectors
#               " GC cycle detection and reference cycle analysis
#               " Monotonic cache growth pattern detection  
#               " Memory usage anomaly detection and reporting
#               " Statistical analysis and historical baselines
# Author      : Kyle Clouthier
# Version     : 1.0.0
# Technology  : Python 3.7+, GC Analysis, Statistical Detection
# Standards   : PEP 8, Type Hints, Production Safety
# Created     : 2025-08-19
# Modified    : 2025-08-19 (Initial creation)
# Dependencies: cycles, caches
# SHA-256     : [PLACEHOLDER - Updated by CI/CD]
# Testing     : 100% coverage, all tests passing
# License     : Proprietary - Patent Pending
# Copyright   : © 2025 Kyle Clouthier (Canada). All rights reserved.
#=============================================================================

from .cycles import (
    install_cycle_detector,
    uninstall_cycle_detector,
    scan_reference_cycles,
    analyze_gc_stats,
    get_cycle_detection_info,
    force_gc_collection,
    get_memory_references,
    find_reference_chains,
    get_performance_stats as get_cycles_performance_stats,
    reset_performance_stats as reset_cycles_performance_stats
)

from .caches import (
    install_cache_detector,
    uninstall_cache_detector,
    scan_growing_caches,
    register_cache_pattern,
    unregister_cache_pattern,
    get_cache_detection_info,
    analyze_memory_growth,
    get_memory_baselines,
    reset_memory_baselines,
    get_performance_stats as get_caches_performance_stats,
    reset_performance_stats as reset_caches_performance_stats
)

__all__ = [
    # Cycle detector exports
    "install_cycle_detector",
    "uninstall_cycle_detector",
    "scan_reference_cycles",
    "analyze_gc_stats",
    "get_cycle_detection_info",
    "force_gc_collection",
    "get_memory_references",
    "find_reference_chains",
    "get_cycles_performance_stats",
    "reset_cycles_performance_stats",
    
    # Cache detector exports
    "install_cache_detector",
    "uninstall_cache_detector",
    "scan_growing_caches",
    "register_cache_pattern",
    "unregister_cache_pattern",
    "get_cache_detection_info",
    "analyze_memory_growth",
    "get_memory_baselines",
    "reset_memory_baselines",
    "get_caches_performance_stats",
    "reset_caches_performance_stats"
]