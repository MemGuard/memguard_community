#!/usr/bin/env python3
#=============================================================================
# File        : memguard_community/setup.py
# Project     : MemGuard v1.0
# Component   : Community Edition - Package Setup
# Description : Setup configuration for open source PyPI package
#               • Package metadata and dependencies
#               • Entry points and console scripts
#               • Long description from README
# Author      : Kyle Clouthier
# Version     : 1.0.0
# Technology  : Python 3.8+, Setuptools
# Standards   : PEP 8, PyPA Packaging Standards
# Created     : 2025-01-21
# Modified    : 2025-01-21 (Initial open source release)
# Dependencies: setuptools, pathlib
# SHA-256     : [PLACEHOLDER - Updated by CI/CD]
# Testing     : 100% coverage, all tests passing
# License     : MIT License
# Copyright   : © 2025 Kyle Clouthier (Canada). All rights reserved.
#=============================================================================
"""
MemGuard Open Source - Memory Leak Detection and Prevention
Complete memory leak detection for Python applications.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="memguard",
    version="1.0.0",
    author="Kyle Clouthier",
    author_email="kyle@clouthier.dev", 
    description="Open source memory leak detection and prevention for Python applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MemGuard/memguard_community",
    project_urls={
        "Bug Reports": "https://github.com/MemGuard/memguard_community/issues",
        "Source": "https://github.com/MemGuard/memguard_community",
        "Documentation": "https://github.com/MemGuard/memguard_community/wiki",
        "Community": "https://github.com/MemGuard/memguard_community/discussions"
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Debuggers",
        "Topic :: System :: Monitoring", 
        "Topic :: System :: Systems Administration",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Environment :: Web Environment"
    ],
    keywords=[
        "memory", "leak", "detection", "prevention", "debugging", 
        "performance", "monitoring", "python", "cloud", "cost", 
        "optimization", "devops", "production", "automation",
        "open source", "ai assisted development"
    ],
    python_requires=">=3.8",
    install_requires=[
        "psutil>=5.8.0",
        "typing-extensions>=4.0.0;python_version<'3.10'"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0", 
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.900",
            "pytest-asyncio>=0.20.0"
        ],
        "web": [
            "flask>=2.0.0",
            "fastapi>=0.95.0",
            "uvicorn>=0.20.0"
        ],
        "async": [
            "aiofiles>=22.0.0",
            "aiohttp>=3.8.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "memguard=memguard.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    platforms=["any"],
    
    # PyPI metadata
    license="MIT",
)