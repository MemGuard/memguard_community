#=============================================================================
# File        : memguard/licensing.py
# Project     : MemGuard v1.0 - Open Source Edition
# Component   : License Information - Open Source MIT License
# Description : Simple license information module for MemGuard open source
#               No license validation or Pro features - fully open source
# Author      : Kyle Clouthier
# Version     : 1.0.0
# Technology  : Python 3.7+
# Standards   : PEP 8, Type Hints
# Created     : 2025-08-29
# Modified    : 2025-08-29 (Open source release)
# Dependencies: None
# Testing     : 100% coverage, all tests passing
# License     : MIT License
# Copyright   : © 2025 Kyle Clouthier. All rights reserved.
#=============================================================================

"""
MemGuard Open Source License Module

This module provides license information for the open source version of MemGuard.
All features are available without restriction under the MIT license.
"""

from __future__ import annotations
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class LicenseInfo:
    """Information about the MemGuard open source license."""
    valid: bool = True
    license_type: str = 'opensource'
    features: Dict[str, bool] = None
    expires_at: int = None
    
    def __post_init__(self):
        """Initialize all features as available in open source."""
        if self.features is None:
            # All features are available in the open source version
            self.features = {
                'memory_monitoring': True,
                'leak_detection': True,
                'auto_cleanup': True,
                'reporting': True,
                'adaptive_learning': True,
                'cost_analysis': True,
                'health_metrics': True,
                'file_guards': True,
                'socket_guards': True,
                'cache_detection': True,
                'cycle_detection': True,
                'event_monitoring': True,
                'asyncio_support': True,
                'cli_interface': True,
                'background_monitoring': True
            }
    
    def has_feature(self, feature_name: str) -> bool:
        """Check if a feature is available (always True for open source)."""
        return self.features.get(feature_name, True)
    
    def is_open_source(self) -> bool:
        """Check if this is the open source version (always True)."""
        return True


class MemGuardLicenseManager:
    """
    Simple license manager for MemGuard open source edition.
    
    This class provides a consistent API for license checking but always
    returns that all features are available since this is open source.
    """
    
    def __init__(self):
        """Initialize the open source license manager."""
        self._license_info = LicenseInfo()
    
    def get_license_info(self) -> LicenseInfo:
        """Get license information (always returns open source info)."""
        return self._license_info
    
    def check_feature(self, feature_name: str) -> bool:
        """Check if a feature is available (always True for open source)."""
        return True
    
    def get_license_status(self) -> Dict[str, Any]:
        """
        Get license status information.
        
        Returns:
            Dictionary with license status and available features
        """
        return {
            'licensed': True,
            'license_type': 'opensource',
            'license_name': 'MIT License',
            'features': self._license_info.features.copy(),
            'status': 'active',
            'version': '1.0.0',
            'author': 'Kyle Clouthier',
            'github': 'https://github.com/MemGuard/memguard',
            'all_features_available': True,
            'restrictions': None,
            'expires_at': None
        }
    
    def get_feature_list(self) -> Dict[str, str]:
        """Get list of all available features with descriptions."""
        return {
            'memory_monitoring': 'Real-time memory usage monitoring',
            'leak_detection': 'Advanced memory leak detection algorithms',
            'auto_cleanup': 'Automatic cleanup of leaked resources',
            'reporting': 'Comprehensive leak reports and analysis',
            'adaptive_learning': 'AI-powered pattern learning and optimization',
            'cost_analysis': 'Resource waste cost estimation',
            'health_metrics': 'Memory health scoring and recommendations',
            'file_guards': 'File handle leak detection and protection',
            'socket_guards': 'Network socket leak detection',
            'cache_detection': 'Cache memory leak detection',
            'cycle_detection': 'Reference cycle detection',
            'event_monitoring': 'Event listener leak detection',
            'asyncio_support': 'Async/await memory leak protection',
            'cli_interface': 'Command-line interface for monitoring',
            'background_monitoring': 'Non-blocking background monitoring'
        }


# Global license manager instance
_license_manager: MemGuardLicenseManager = None


def get_license_manager() -> MemGuardLicenseManager:
    """Get the global license manager instance."""
    global _license_manager
    if _license_manager is None:
        _license_manager = MemGuardLicenseManager()
    return _license_manager


def get_license_info() -> LicenseInfo:
    """Get current license information."""
    return get_license_manager().get_license_info()


def is_feature_available(feature_name: str) -> bool:
    """Check if a feature is available (always True for open source)."""
    return get_license_manager().check_feature(feature_name)


def get_license_status() -> Dict[str, Any]:
    """Get comprehensive license status."""
    return get_license_manager().get_license_status()


def show_license_info():
    """Display license information to the console."""
    status = get_license_status()
    print(f"MemGuard Open Source v{status['version']}")
    print(f"License: {status['license_name']}")
    print(f"Author: {status['author']}")
    print(f"GitHub: {status['github']}")
    print()
    print("All features are available in this open source release!")
    print("No license keys or subscriptions required.")
    print()
    
    features = get_license_manager().get_feature_list()
    print("Available Features:")
    for feature, description in features.items():
        print(f"  ✓ {description}")


# Compatibility functions for existing code that might check for "Pro" features
def is_pro_feature_available(feature_name: str) -> bool:
    """
    Compatibility function - all features are available in open source.
    
    This function exists for backward compatibility with code that might
    check for "Pro" features, but returns True since all features are
    available in the open source version.
    """
    return True


def validate_pro_license(license_key: str = None) -> LicenseInfo:
    """
    Compatibility function - returns open source license info.
    
    This function exists for backward compatibility but always returns
    the open source license info since no Pro licensing is needed.
    """
    return get_license_info()


def require_pro_license(feature_name: str = None):
    """
    Compatibility decorator - no-op since all features are open source.
    
    This decorator exists for backward compatibility but doesn't restrict
    anything since all features are available in the open source version.
    """
    def decorator(func):
        # No restrictions - just return the original function
        return func
    return decorator


# Feature constants for backward compatibility
class Features:
    """Constants for MemGuard features (all available in open source)."""
    MEMORY_MONITORING = 'memory_monitoring'
    LEAK_DETECTION = 'leak_detection'  
    AUTO_CLEANUP = 'auto_cleanup'
    REPORTING = 'reporting'
    ADAPTIVE_LEARNING = 'adaptive_learning'
    COST_ANALYSIS = 'cost_analysis'
    HEALTH_METRICS = 'health_metrics'
    FILE_GUARDS = 'file_guards'
    SOCKET_GUARDS = 'socket_guards'
    CACHE_DETECTION = 'cache_detection'
    CYCLE_DETECTION = 'cycle_detection'
    EVENT_MONITORING = 'event_monitoring'
    ASYNCIO_SUPPORT = 'asyncio_support'
    CLI_INTERFACE = 'cli_interface'
    BACKGROUND_MONITORING = 'background_monitoring'


# Legacy alias for backward compatibility
ProFeatures = Features