# 📦 Installation Guide

## Quick Install

### From GitHub (Recommended)

```bash
# Clone the repository
git clone https://github.com/MemGuard/memguard_community.git
cd memguard_community

# Install in development mode
pip install -e .

# Or install with all features
pip install -e ".[dev,web,async]"
```

### From PyPI (Coming Soon)

```bash
# Install the latest release
pip install memguard

# Install with optional dependencies
pip install memguard[web,async]
```

## Requirements

- **Python**: 3.8+ (3.9+ recommended)
- **Operating System**: Linux, macOS, Windows
- **Memory**: Minimal impact (<50MB overhead)
- **Dependencies**: Listed in `requirements.txt`

## Installation Options

### Basic Installation
```bash
pip install -e .
```
Includes core monitoring and cleanup features.

### Development Installation
```bash
pip install -e ".[dev]"
```
Adds testing tools: `pytest`, `black`, `flake8`, `mypy`

### Web Application Support
```bash
pip install -e ".[web]"
```
Adds Flask, FastAPI, and Uvicorn for web demos.

### Async Application Support
```bash
pip install -e ".[async]"
```
Adds aiofiles and aiohttp for async monitoring.

### Complete Installation
```bash
pip install -e ".[dev,web,async]"
```
Installs all optional dependencies.

## Verify Installation

```python
import memguard
print(memguard.get_info())
```

Expected output:
```
🛡️ MemGuard v1.0.0 - Open Source Memory Leak Detection

Built by: Kyle Clouthier
License: MIT
Repository: https://github.com/MemGuard/memguard_community
...
```

## Quick Test

```python
import memguard
import time

# Start monitoring
memguard.protect()

# Create some test activity
for i in range(10):
    with open(f'test_{i}.tmp', 'w') as f:
        f.write('test data')
    time.sleep(0.1)

# Check status
status = memguard.get_status()
print(f"Overhead: {status['performance_stats']['overhead_percentage']:.2f}%")
print(f"Scans performed: {status['scan_count']}")

# Cleanup
memguard.stop()
```

## Troubleshooting

### Import Errors
```bash
# Ensure you're in the right directory
cd memguard_community
python -c "import memguard"
```

### Permission Issues
```bash
# Install in user directory
pip install --user -e .
```

### Python Version Issues
```bash
# Check Python version
python --version

# Use specific Python version
python3.9 -m pip install -e .
```

### Missing Dependencies
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3-dev

# Install system dependencies (macOS)
brew install python@3.9

# Install system dependencies (Windows)
# Use Python installer from python.org
```

## Development Setup

```bash
# Clone repository
git clone https://github.com/MemGuard/memguard_community.git
cd memguard_community

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install development version
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
black memguard/
flake8 memguard/
mypy memguard/
```

## Docker Installation (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -e .

CMD ["python", "-c", "import memguard; memguard.protect()"]
```

## Next Steps

- **[Quick Start Guide](README.md#quick-start)** - Basic usage examples
- **[Configuration Guide](docs/configuration.md)** - Advanced settings  
- **[Integration Examples](docs/integrations.md)** - Framework-specific guides
- **[Performance Testing](tests/)** - Validate installation with real metrics

## Support

- **Issues**: [Report problems](https://github.com/MemGuard/memguard_community/issues)
- **Discussions**: [Ask questions](https://github.com/MemGuard/memguard_community/discussions)
- **Documentation**: [Complete guides](https://github.com/MemGuard/memguard_community/wiki)