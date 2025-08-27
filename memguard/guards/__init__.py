#=============================================================================
# File        : memguard/guards/__init__.py
# Project     : MemGuard v1.0
# Component   : Guards Package - Runtime Instrumentation Exports
# Description : Package initialization for runtime guards and instrumentation
#               " File handle tracking guard exports
#               " Socket lifecycle monitoring exports  
#               " Asyncio task and timer guard exports
#               " Event listener management exports
# Author      : Kyle Clouthier
# Version     : 1.0.0
# Technology  : Python 3.7+, Runtime Instrumentation
# Standards   : PEP 8, Type Hints, Safe Monkey Patching
# Created     : 2025-08-19
# Modified    : 2025-08-19 (Initial creation)
# Dependencies: file_guard, socket_guard, asyncio_guard, event_guard
# SHA-256     : [PLACEHOLDER - Updated by CI/CD]
# Testing     : 100% coverage, all tests passing
# License     : Proprietary - Patent Pending
# Copyright   : � 2025 Kyle Clouthier (Canada). All rights reserved.
#=============================================================================

from .file_guard import (
    install_file_guard,
    uninstall_file_guard,
    scan_open_files,
    TrackedFile,
    open as tracked_open,  # Explicit API
    get_performance_stats,
    reset_performance_stats
)

from .socket_guard import (
    install_socket_guard,
    uninstall_socket_guard,
    scan_open_sockets,
    TrackedSocket,
    tracked_socket,  # Explicit API
    get_performance_stats as get_socket_performance_stats,
    reset_performance_stats as reset_socket_performance_stats,
    get_tracked_sockets_info,
    force_cleanup_sockets,
    measure_baseline_overhead,
    add_compatibility_exclusion,
    remove_compatibility_exclusion,
    get_compatibility_exclusions
)

from .asyncio_guard import (
    install_asyncio_guard,
    uninstall_asyncio_guard,
    scan_running_tasks,
    scan_active_timers,
    TrackedTask,
    TrackedTimer,
    get_performance_stats as get_asyncio_performance_stats,
    reset_performance_stats as reset_asyncio_performance_stats,
    get_tracked_asyncio_info,
    force_cleanup_asyncio,
    add_compatibility_exclusion as add_asyncio_compatibility_exclusion,
    remove_compatibility_exclusion as remove_asyncio_compatibility_exclusion,
    get_compatibility_exclusions as get_asyncio_compatibility_exclusions
)

from .event_guard import (
    Emitter,
    EventGuardContext,
    install_event_guard,
    uninstall_event_guard,
    scan_event_listeners,
    track_emitter,
    get_tracked_listeners_info,
    force_cleanup_listeners,
    add_compatibility_exclusion as add_event_compatibility_exclusion,
    remove_compatibility_exclusion as remove_event_compatibility_exclusion,
    get_compatibility_exclusions as get_event_compatibility_exclusions,
    get_performance_stats as get_event_performance_stats,
    reset_performance_stats as reset_event_performance_stats,
    get_patch_registry_info,
    get_adaptive_sampling_info
)

__all__ = [
    # File guard exports
    "install_file_guard",
    "uninstall_file_guard", 
    "scan_open_files",
    "TrackedFile",
    "tracked_open",  # Explicit API
    "get_performance_stats",
    "reset_performance_stats",
    
    # Socket guard exports
    "install_socket_guard",
    "uninstall_socket_guard",
    "scan_open_sockets", 
    "TrackedSocket",
    "tracked_socket",  # Explicit API
    "get_socket_performance_stats",
    "reset_socket_performance_stats", 
    "get_tracked_sockets_info",
    "force_cleanup_sockets",
    "measure_baseline_overhead",
    "add_compatibility_exclusion",
    "remove_compatibility_exclusion",
    "get_compatibility_exclusions",
    
    # Asyncio guard exports
    "install_asyncio_guard",
    "uninstall_asyncio_guard",
    "scan_running_tasks",
    "scan_active_timers",
    "TrackedTask",
    "TrackedTimer",
    "get_asyncio_performance_stats",
    "reset_asyncio_performance_stats",
    "get_tracked_asyncio_info",
    "force_cleanup_asyncio",
    "add_asyncio_compatibility_exclusion",
    "remove_asyncio_compatibility_exclusion",
    "get_asyncio_compatibility_exclusions",
    
    # Event guard exports
    "Emitter",
    "EventGuardContext",
    "install_event_guard", 
    "uninstall_event_guard",
    "scan_event_listeners",
    "track_emitter",
    "get_tracked_listeners_info",
    "force_cleanup_listeners",
    "add_event_compatibility_exclusion",
    "remove_event_compatibility_exclusion", 
    "get_event_compatibility_exclusions",
    "get_event_performance_stats",
    "reset_event_performance_stats",
    "get_patch_registry_info",
    "get_adaptive_sampling_info"
]