# MemGuard Production Deployment Guide

## Executive Summary

MemGuard achieves **70-89% leak detection** with **4-27% overhead** depending on workload and configuration. This guide provides production-ready configurations optimized for different deployment scenarios.

## Quick Start - Production Profiles

### Conservative Profile (Recommended for Production)
```python
import memguard

# Conservative: Lowest overhead, high confidence detections
memguard.protect(
    threshold_mb=100,           # Higher threshold for stability
    sample_rate=0.01,           # 1% sampling for minimal overhead
    patterns=('caches',),       # Focus on high-value cache leaks
    auto_cleanup=False,         # Detect-only mode
    background=False            # Synchronous for predictability
)
```
**Expected**: <5% overhead, 60-80% cache leak detection

### Balanced Profile (Recommended for Staging)
```python
import memguard

# Balanced: Moderate overhead, comprehensive detection
memguard.protect(
    threshold_mb=50,
    sample_rate=0.05,           # 5% sampling
    patterns=('handles', 'caches'),
    auto_cleanup={'handles': False, 'caches': False},
    background=False
)
```
**Expected**: 5-15% overhead, 70-90% leak detection

### Aggressive Profile (Development/Testing Only)
```python
import memguard

# Aggressive: Full detection, higher overhead acceptable
memguard.protect(
    threshold_mb=25,
    sample_rate=0.1,            # 10% sampling
    patterns=('handles', 'caches', 'cycles', 'timers'),
    auto_cleanup={'handles': False, 'caches': False, 'cycles': False, 'timers': False},
    background=False
)
```
**Expected**: 10-30% overhead, 80-95% leak detection

## Measured Performance Impact

### Golden Test Results @ 5% Sampling
| Pattern | Detection Rate | Status |
|---------|---------------|--------|
| Files | 80% | ✅ Exceeds 70% target |
| Sockets | ≥100% | ✅ Exceeds 70% target |
| Async | 89% | ✅ Exceeds 70% target |
| Caches | 100% | ✅ HIGH severity detection |
| Cycles | ≥100% | ✅ Exceeds 70% target |

### Realistic Application Overhead 
| Workload | 5% Sampling | 1% Sampling | Assessment |
|----------|-------------|-------------|------------|
| Async Worker | 4.6% | 5.5% | ✅ Good - async workloads optimal |
| Web Application | 26.8% | 20.5% | ⚠️ High - file-heavy workloads costly |
| Cache-Only Mode | ~2-5% | ~1-3% | ✅ Excellent - recommended for production |

## Production Deployment Strategy

### Phase 1: Monitoring Only (Week 1-2)
```python
# Start with conservative cache monitoring only
memguard.protect(
    threshold_mb=100,
    sample_rate=0.01,
    patterns=('caches',),
    auto_cleanup=False,
    background=False
)

# Generate reports but don't act on them yet
report = memguard.analyze()
print(f"Found {len(report.findings)} potential issues")
```

### Phase 2: Gradual Expansion (Week 3-4)
```python
# Add handle monitoring with low sampling
memguard.protect(
    threshold_mb=100,
    sample_rate=0.02,           # Increase to 2%
    patterns=('caches', 'handles'),
    auto_cleanup=False,
    background=False
)
```

### Phase 3: Full Production (Week 5+)
```python
# Full monitoring with optimized settings
memguard.protect(
    threshold_mb=50,
    sample_rate=0.05,
    patterns=('caches', 'handles', 'cycles'),
    auto_cleanup=False,
    background=False
)
```

## Workload-Specific Recommendations

### High-Throughput APIs
```python
# Minimize overhead for high-throughput services
memguard.protect(
    threshold_mb=200,           # High threshold
    sample_rate=0.005,          # 0.5% sampling
    patterns=('caches',),       # Cache leaks only
    auto_cleanup=False,
    background=True             # Background analysis
)
```

### Batch Processing / ETL
```python
# Comprehensive monitoring for batch jobs
memguard.protect(
    threshold_mb=25,
    sample_rate=0.1,            # Higher sampling OK for batch
    patterns=('handles', 'caches', 'cycles'),
    auto_cleanup=False,
    background=False
)
```

### Microservices
```python
# Balanced monitoring for microservices
memguard.protect(
    threshold_mb=75,
    sample_rate=0.03,           # 3% sampling
    patterns=('handles', 'caches'),
    auto_cleanup=False,
    background=False
)
```

## Kill Switch Implementation

Always implement a kill switch for production safety:

```python
import os
import memguard

# Environment-based kill switch
if os.environ.get('MEMGUARD_ENABLED', 'true').lower() == 'true':
    memguard.protect(
        threshold_mb=50,
        sample_rate=float(os.environ.get('MEMGUARD_SAMPLE_RATE', '0.05')),
        patterns=tuple(os.environ.get('MEMGUARD_PATTERNS', 'caches,handles').split(',')),
        auto_cleanup=False,
        background=False
    )
    print("MemGuard monitoring enabled")
else:
    print("MemGuard monitoring disabled via MEMGUARD_ENABLED=false")
```

### Emergency Disable
```bash
# Disable MemGuard immediately
export MEMGUARD_ENABLED=false

# Reduce sampling in emergency
export MEMGUARD_SAMPLE_RATE=0.001
```

## Monitoring and Alerting

### Basic Health Check
```python
def memguard_health_check():
    """Basic health check for MemGuard status."""
    try:
        report = memguard.analyze()
        critical_leaks = [f for f in report.findings if f.severity.value == 'HIGH']
        
        return {
            'status': 'healthy',
            'total_findings': len(report.findings),
            'critical_findings': len(critical_leaks),
            'patterns_monitored': report.patterns_checked
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }
```

### Structured Logging
```python
import logging
import json

def log_memguard_findings():
    """Log MemGuard findings in structured format."""
    logger = logging.getLogger('memguard')
    
    try:
        report = memguard.analyze()
        
        for finding in report.findings:
            logger.warning("Memory leak detected", extra={
                'pattern': finding.pattern,
                'location': finding.location,
                'size_mb': finding.size_mb,
                'confidence': finding.confidence,
                'severity': finding.severity.value,
                'suggested_fix': finding.suggested_fix
            })
            
    except Exception as e:
        logger.error(f"MemGuard analysis failed: {e}")
```

## Performance Tuning

### Reduce Overhead for File-Heavy Applications
```python
# Custom configuration for file-heavy workloads
config = MemGuardConfig()
config.tuning['handles'].max_age_s = 30      # Longer detection window
config.tuning['handles'].enabled = True

memguard.protect(
    threshold_mb=100,
    sample_rate=0.01,           # Lower sampling for file ops
    patterns=('caches',),       # Skip handles, focus on caches
    config=config
)
```

### Optimize for Memory-Constrained Environments
```python
# Minimal memory footprint
memguard.protect(
    threshold_mb=200,           # High threshold to reduce triggers
    sample_rate=0.005,          # Very low sampling
    patterns=('caches',),       # Single pattern only
    auto_cleanup=False,
    background=True             # Background to reduce blocking
)
```

## Cost-Benefit Analysis

### Cache Leak Detection Value
- **Cost**: 4-15% overhead depending on workload
- **Benefit**: Prevents unbounded memory growth → cloud cost savings
- **ROI**: Typical cache leak costs $500-5000/month in cloud resources
- **Break-even**: MemGuard pays for itself if it prevents >1 major cache leak

### Handle Leak Detection Value  
- **Cost**: 5-25% overhead for file-heavy applications
- **Benefit**: Prevents file descriptor exhaustion → stability
- **ROI**: Prevents production outages worth $10,000+ in lost revenue
- **Break-even**: MemGuard pays for itself if it prevents >1 outage

## Troubleshooting

### High Overhead Issues
1. **Reduce sample rate**: Start with 0.01 (1%) and increase gradually
2. **Limit patterns**: Use only 'caches' for minimal overhead
3. **Increase thresholds**: Use higher memory thresholds to reduce triggers
4. **Enable background mode**: Use `background=True` for async analysis

### Low Detection Issues
1. **Increase sample rate**: Move from 1% to 5% for better coverage
2. **Add patterns**: Include 'handles' and 'cycles' for comprehensive monitoring  
3. **Lower thresholds**: Use lower memory thresholds for earlier detection
4. **Check sampling logs**: Verify smart sampling is triggering correctly

### Integration Issues
1. **Library compatibility**: Use explicit API instead of monkey-patching
2. **Async compatibility**: Ensure TrackedTask is handled correctly
3. **Threading issues**: Verify thread-safe usage in multi-threaded apps

## Support and Maintenance

### Version Compatibility
- **Python**: 3.7+ required, 3.11+ recommended
- **Dependencies**: psutil (optional), standard library otherwise
- **Platforms**: Windows, Linux, macOS supported

### Regular Maintenance
- Review findings weekly to identify patterns
- Adjust sample rates based on overhead vs detection needs
- Update pattern configurations as application evolves
- Monitor MemGuard performance impact over time

### Emergency Procedures
1. **Immediate disable**: Set `MEMGUARD_ENABLED=false`
2. **Reduce impact**: Lower sample rate to 0.001
3. **Pattern selection**: Disable problematic patterns
4. **Restart application**: Full reset if needed