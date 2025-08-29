#!/usr/bin/env python3
"""
MemGuard Open Source Setup Configuration
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "MemGuard - Open Source Memory Leak Detection and Auto-Cleanup"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Read version from memguard/__init__.py
def get_version():
    version_path = os.path.join(os.path.dirname(__file__), 'memguard', '__init__.py')
    if os.path.exists(version_path):
        with open(version_path, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return '1.0.0'

setup(
    name='memguard',
    version=get_version(),
    author='Kyle Clouthier',
    author_email='kyle@example.com',
    description='Open Source Memory Leak Detection and Auto-Cleanup for Python Applications',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/MemGuard/memguard_community',
    project_urls={
        'Bug Reports': 'https://github.com/MemGuard/memguard_community/issues',
        'Source': 'https://github.com/MemGuard/memguard_community',
        'Documentation': 'https://github.com/MemGuard/memguard_community/wiki',
    },
    packages=find_packages(exclude=['tests*', 'docs*', 'examples*']),
    classifiers=[
        # Development Status
        'Development Status :: 5 - Production/Stable',
        
        # Intended Audience
        'Intended Audience :: Developers',
        'Intended Audience :: System Administrators',
        
        # Topic
        'Topic :: Software Development :: Debuggers',
        'Topic :: Software Development :: Quality Assurance',
        'Topic :: System :: Monitoring',
        'Topic :: System :: Systems Administration',
        
        # License
        'License :: OSI Approved :: MIT License',
        
        # Python Versions
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        
        # Operating Systems
        'Operating System :: OS Independent',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
    ],
    python_requires='>=3.8',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=1.0.0',
        ],
        'web': [
            'flask>=2.0.0',
            'fastapi>=0.95.0',
            'uvicorn>=0.20.0',
        ],
        'async': [
            'aiofiles>=22.0.0',
            'aiohttp>=3.8.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'memguard=memguard.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        'memguard': [
            'config/*.yaml',
            'config/*.json',
            'templates/*.html',
            'static/*.css',
            'static/*.js',
        ],
    },
    zip_safe=False,
    keywords=[
        'memory leak detection',
        'memory profiling',
        'resource monitoring',
        'auto cleanup',
        'garbage collection',
        'performance monitoring',
        'debugging tools',
        'production monitoring',
        'memory management',
        'resource tracking'
    ],
)