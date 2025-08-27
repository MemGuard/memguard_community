# MemGuard Test Organization

## Directory Structure

- `validation/` - Key validation tests for website and documentation
- `benchmarks/` - Performance comparison tests  
- `results/` - Test results and reports
- `archive/` - Historical tests no longer needed

## Key Tests for Website

### Primary Validation Tests
1. **test_clean_normal_mode.py** - Main production readiness test
2. **test_pro_features_real.py** - Pro features validation
3. **test_license.py** - License system validation

### Performance Benchmarks  
1. **test_aggressive_mode_performance.py** - Mode comparison (archived - aggressive removed)
2. **test_auto_cleanup_optimized.py** - Enhanced performance test

### Archive (Historical)
- All 4-hour test variations (abandoned due to GC debug issues)
- Aggressive mode tests (feature removed)
- Debug/development tests

## Website-Ready Results

Use results from:
- `validation/test_clean_normal_mode.py` - Shows production performance
- `results/enhanced_test_results_2025_08_21.md` - Enhanced test results
- Pro feature demonstrations

## Test Status Legend
- ✅ Production ready / Website worthy
- 🔧 Development/debugging  
- 📦 Archived/historical
- ❌ Deprecated/removed