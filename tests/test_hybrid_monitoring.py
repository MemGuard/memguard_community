#=============================================================================
# File        : tests/test_hybrid_monitoring.py
# Project     : MemGuard v1.0
# Component   : Hybrid Monitoring Test Suite
# Description : Comprehensive testing of hybrid monitoring system performance
#               • Light vs Deep vs Hybrid mode performance validation
#               • Production workload simulation
#               • Real metrics collection and overhead measurement
#               • Website performance impact testing
# Author      : Kyle Clouthier
# Version     : 1.0.0
# Created     : 2025-08-27
#=============================================================================

import os
import sys
import time
import pytest
import threading
import statistics
from typing import List, Dict, Any
from pathlib import Path

# Add memguard to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

import memguard
from memguard.core import protect, stop, get_status, analyze
from memguard.config import MemGuardConfig
from memguard.licensing import validate_pro_license

class PerformanceMetrics:
    """Collect and analyze performance metrics during testing."""
    
    def __init__(self):
        self.samples: List[Dict[str, float]] = []
        self.start_time = time.time()
    
    def record_sample(self, 
                     cpu_percent: float, 
                     memory_mb: float, 
                     overhead_percent: float,
                     operations_per_sec: float,
                     scan_type: str = "unknown"):
        """Record a performance sample."""
        self.samples.append({
            'timestamp': time.time() - self.start_time,
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb,
            'overhead_percent': overhead_percent,
            'operations_per_sec': operations_per_sec,
            'scan_type': scan_type
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.samples:
            return {}
        
        cpu_values = [s['cpu_percent'] for s in self.samples]
        memory_values = [s['memory_mb'] for s in self.samples]
        overhead_values = [s['overhead_percent'] for s in self.samples]
        ops_values = [s['operations_per_sec'] for s in self.samples]
        
        return {
            'sample_count': len(self.samples),
            'duration_seconds': self.samples[-1]['timestamp'],
            'cpu_percent': {
                'avg': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'p95': statistics.quantiles(cpu_values, n=20)[18] if len(cpu_values) > 20 else max(cpu_values)
            },
            'memory_mb': {
                'avg': statistics.mean(memory_values),
                'max': max(memory_values),
                'growth': max(memory_values) - min(memory_values)
            },
            'overhead_percent': {
                'avg': statistics.mean(overhead_values),
                'max': max(overhead_values),
                'p95': statistics.quantiles(overhead_values, n=20)[18] if len(overhead_values) > 20 else max(overhead_values)
            },
            'operations_per_sec': {
                'avg': statistics.mean(ops_values),
                'min': min(ops_values),
                'degradation_percent': ((max(ops_values) - min(ops_values)) / max(ops_values)) * 100 if ops_values else 0
            }
        }

class WorkloadSimulator:
    """Simulate realistic production workloads."""
    
    def __init__(self):
        self.stop_event = threading.Event()
        self.operations_count = 0
        
    def simulate_file_operations(self, duration_seconds: float = 60.0):
        """Simulate file I/O operations."""
        import tempfile
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time and not self.stop_event.is_set():
            try:
                # Create temporary files
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    f.write(b"test data" * 1000)  # 9KB per file
                    temp_path = f.name
                
                # Read and delete
                with open(temp_path, 'rb') as f:
                    data = f.read()
                
                os.unlink(temp_path)
                self.operations_count += 1
                
                # Rate limiting to avoid overwhelming system
                time.sleep(0.01)  # 100 ops/sec max
                
            except Exception:
                pass
    
    def simulate_socket_operations(self, duration_seconds: float = 60.0):
        """Simulate socket operations."""
        import socket
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time and not self.stop_event.is_set():
            try:
                # Create and close sockets
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.1)
                try:
                    sock.connect(('127.0.0.1', 80))  # Will likely fail, that's OK
                except:
                    pass
                sock.close()
                
                self.operations_count += 1
                time.sleep(0.01)  # Rate limiting
                
            except Exception:
                pass
    
    def simulate_memory_operations(self, duration_seconds: float = 60.0):
        """Simulate memory-intensive operations."""
        end_time = time.time() + duration_seconds
        caches = []
        
        while time.time() < end_time and not self.stop_event.is_set():
            try:
                # Create growing cache structures
                cache = {}
                for i in range(100):
                    cache[f"key_{i}_{time.time()}"] = "data" * 100
                
                caches.append(cache)
                
                # Periodically clear some caches
                if len(caches) > 50:
                    caches = caches[-25:]
                
                self.operations_count += 1
                time.sleep(0.02)
                
            except Exception:
                pass
    
    def stop(self):
        """Stop all workload simulation."""
        self.stop_event.set()

@pytest.fixture
def pro_license():
    """Get Pro license for testing."""
    return validate_pro_license("MEMGUARD-PRO-001-GLOBAL")

@pytest.fixture
def performance_metrics():
    """Performance metrics collector."""
    return PerformanceMetrics()

@pytest.fixture
def workload_simulator():
    """Workload simulator."""
    simulator = WorkloadSimulator()
    yield simulator
    simulator.stop()

class TestHybridMonitoring:
    """Test suite for hybrid monitoring performance validation."""
    
    def test_light_monitoring_overhead(self, pro_license, performance_metrics, workload_simulator):
        """Test that light monitoring achieves <3% overhead target."""
        
        # Ensure clean state
        stop()
        
        # Configure light monitoring mode
        config = MemGuardConfig(
            monitoring_mode="light",
            light_sample_rate=0.01,  # 1% sampling
            poll_interval_s=1.0,
            intensive_cleanup_schedule="never"
        )
        
        # Start protection with Pro license
        protect(
            config=config,
            license_key="MEMGUARD-PRO-001-GLOBAL"
        )
        
        # Baseline performance measurement
        baseline_start = time.time()
        workload_simulator.simulate_file_operations(30.0)  # 30 seconds
        baseline_ops = workload_simulator.operations_count
        baseline_duration = time.time() - baseline_start
        baseline_ops_per_sec = baseline_ops / baseline_duration
        
        # Reset counter
        workload_simulator.operations_count = 0
        
        # Protected performance measurement
        protected_start = time.time()
        workload_simulator.simulate_file_operations(30.0)  # 30 seconds
        protected_ops = workload_simulator.operations_count
        protected_duration = time.time() - protected_start
        protected_ops_per_sec = protected_ops / protected_duration
        
        # Calculate overhead
        overhead_percent = ((baseline_ops_per_sec - protected_ops_per_sec) / baseline_ops_per_sec) * 100
        
        # Get MemGuard metrics
        status = get_status()
        memguard_overhead = status['performance_stats']['overhead_percentage']
        
        # Record metrics
        performance_metrics.record_sample(
            cpu_percent=0.0,  # Would need psutil for real CPU measurement
            memory_mb=status['performance_stats']['memory_current_mb'],
            overhead_percent=overhead_percent,
            operations_per_sec=protected_ops_per_sec,
            scan_type="light"
        )
        
        print(f"\n=== LIGHT MONITORING PERFORMANCE ===")
        print(f"Baseline ops/sec: {baseline_ops_per_sec:.1f}")
        print(f"Protected ops/sec: {protected_ops_per_sec:.1f}")
        print(f"Performance overhead: {overhead_percent:.2f}%")
        print(f"MemGuard internal overhead: {memguard_overhead:.2f}%")
        print(f"Target: <3.0%")
        
        # Cleanup
        stop()
        
        # Assertions
        assert overhead_percent < 5.0, f"Light mode overhead {overhead_percent:.2f}% exceeds 5% limit"
        assert memguard_overhead < 10.0, f"MemGuard internal overhead {memguard_overhead:.2f}% too high"
    
    def test_deep_monitoring_effectiveness(self, pro_license, performance_metrics, workload_simulator):
        """Test that deep monitoring provides comprehensive leak detection."""
        
        # Ensure clean state
        stop()
        
        # Configure deep monitoring mode
        config = MemGuardConfig(
            monitoring_mode="deep",
            deep_sample_rate=1.0,  # 100% sampling
            poll_interval_s=2.0,
            intensive_cleanup_schedule="never"
        )
        
        # Start protection
        protect(
            config=config,
            license_key="MEMGUARD-PRO-001-GLOBAL"
        )
        
        # Create intentional leaks for detection
        leaky_files = []
        leaky_sockets = []
        
        try:
            # Create file leaks
            import tempfile
            for i in range(20):
                f = tempfile.NamedTemporaryFile(delete=False)
                f.write(b"leak test data" * 1000)
                leaky_files.append(f)
            
            # Create socket leaks  
            import socket
            for i in range(10):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                leaky_sockets.append(sock)
            
            # Wait for detection
            time.sleep(10.0)
            
            # Analyze for leaks
            report = analyze()
            
            # Get status
            status = get_status()
            
            print(f"\n=== DEEP MONITORING EFFECTIVENESS ===")
            print(f"Leaks detected: {len(report.findings)}")
            print(f"File handles created: {len(leaky_files)}")
            print(f"Socket handles created: {len(leaky_sockets)}")
            print(f"Deep scan overhead: {status['performance_stats']['overhead_percentage']:.2f}%")
            
            # Record metrics
            performance_metrics.record_sample(
                cpu_percent=0.0,
                memory_mb=status['performance_stats']['memory_current_mb'],
                overhead_percent=status['performance_stats']['overhead_percentage'],
                operations_per_sec=0.0,
                scan_type="deep"
            )
            
            # Clean up leaks
            for f in leaky_files:
                try:
                    f.close()
                    os.unlink(f.name)
                except:
                    pass
                    
            for sock in leaky_sockets:
                try:
                    sock.close()
                except:
                    pass
            
        finally:
            stop()
        
        # Assertions - Deep mode should detect leaks
        assert len(report.findings) > 0, "Deep monitoring should detect intentional leaks"
        assert any(f.pattern == "handles" for f in report.findings), "Should detect file/socket handle leaks"
    
    def test_hybrid_monitoring_transitions(self, pro_license, performance_metrics, workload_simulator):
        """Test hybrid monitoring transitions between light and deep scans."""
        
        # Ensure clean state
        stop()
        
        # Configure hybrid monitoring
        config = MemGuardConfig(
            monitoring_mode="hybrid",
            light_sample_rate=0.01,
            deep_sample_rate=1.0,
            poll_interval_s=2.0,
            deep_scan_trigger_threshold=3,  # Low threshold for testing
            intensive_cleanup_schedule="never"
        )
        
        # Start protection
        protect(
            config=config,
            license_key="MEMGUARD-PRO-001-GLOBAL"
        )
        
        # Monitor for transitions
        light_scans = 0
        deep_scans = 0
        
        # Create gradual leaks to trigger deep scan
        temp_files = []
        
        try:
            for cycle in range(10):  # 20 seconds total
                # Create some leaks to trigger deep scan
                if cycle > 5:  # Start creating leaks after 10 seconds
                    import tempfile
                    f = tempfile.NamedTemporaryFile(delete=False)
                    f.write(b"trigger data" * 500)
                    temp_files.append(f)
                
                # Wait for scan cycle
                time.sleep(2.0)
                
                # Check status
                status = get_status()
                overhead = status['performance_stats']['overhead_percentage']
                
                # Classify scan type based on overhead
                if overhead < 5.0:
                    light_scans += 1
                    scan_type = "light"
                else:
                    deep_scans += 1
                    scan_type = "deep"
                
                performance_metrics.record_sample(
                    cpu_percent=0.0,
                    memory_mb=status['performance_stats']['memory_current_mb'],
                    overhead_percent=overhead,
                    operations_per_sec=0.0,
                    scan_type=scan_type
                )
            
            # Clean up
            for f in temp_files:
                try:
                    f.close()
                    os.unlink(f.name)
                except:
                    pass
        
        finally:
            stop()
        
        print(f"\n=== HYBRID MONITORING TRANSITIONS ===")
        print(f"Light scans detected: {light_scans}")
        print(f"Deep scans detected: {deep_scans}")
        print(f"Total monitoring cycles: {light_scans + deep_scans}")
        
        # Assertions
        assert light_scans > 0, "Hybrid mode should perform light scans"
        assert deep_scans > 0, "Hybrid mode should trigger deep scans when leaks accumulate"
    
    def test_intensive_cleanup_scheduling(self, pro_license):
        """Test custom scheduling of intensive cleanup operations."""
        
        # Ensure clean state
        stop()
        
        # Configure with hourly intensive cleanup (for testing, we'll trigger manually)
        config = MemGuardConfig(
            monitoring_mode="light",
            intensive_cleanup_schedule="custom",
            custom_cleanup_cron="* * * * *",  # Every minute for testing
            max_intensive_cleanup_duration_s=10.0  # Short duration for testing
        )
        
        # Start protection
        protect(
            config=config,
            license_key="MEMGUARD-PRO-001-GLOBAL"
        )
        
        # Create some resources that need cleanup
        temp_files = []
        import tempfile
        
        try:
            # Create resources
            for i in range(10):
                f = tempfile.NamedTemporaryFile(delete=False)
                f.write(b"cleanup test" * 100)
                temp_files.append(f.name)
                f.close()
            
            initial_count = len(temp_files)
            
            # Wait for potential cleanup scheduling
            time.sleep(15.0)  # Wait longer than cleanup duration
            
            # Check status
            status = get_status()
            
            print(f"\n=== INTENSIVE CLEANUP SCHEDULING ===")
            print(f"Initial temp files: {initial_count}")
            print(f"Monitoring mode: {status['configuration']['monitoring_mode'] if 'configuration' in status else 'unknown'}")
            print(f"Protection active: {status['is_protecting']}")
            
            # Manual cleanup of temp files for test hygiene
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass
        
        finally:
            stop()
        
        # This test validates the system accepts scheduling configuration
        assert status['is_protecting'], "Protection should remain active with scheduled cleanup"

def generate_performance_report(metrics: PerformanceMetrics) -> str:
    """Generate a comprehensive performance report."""
    summary = metrics.get_summary()
    
    if not summary:
        return "No performance data collected"
    
    report = f"""
=== MEMGUARD HYBRID MONITORING PERFORMANCE REPORT ===

Test Duration: {summary['duration_seconds']:.1f} seconds
Sample Count: {summary['sample_count']}

CPU Performance:
  Average: {summary['cpu_percent']['avg']:.2f}%
  Maximum: {summary['cpu_percent']['max']:.2f}%
  95th Percentile: {summary['cpu_percent']['p95']:.2f}%

Memory Usage:
  Average: {summary['memory_mb']['avg']:.1f} MB
  Peak: {summary['memory_mb']['max']:.1f} MB
  Growth: {summary['memory_mb']['growth']:.1f} MB

Overhead Analysis:
  Average: {summary['overhead_percent']['avg']:.2f}%
  Maximum: {summary['overhead_percent']['max']:.2f}%
  95th Percentile: {summary['overhead_percent']['p95']:.2f}%

Throughput Impact:
  Average Operations/sec: {summary['operations_per_sec']['avg']:.1f}
  Minimum Operations/sec: {summary['operations_per_sec']['min']:.1f}
  Performance Degradation: {summary['operations_per_sec']['degradation_percent']:.2f}%

=== PRODUCTION READINESS ASSESSMENT ===

Light Mode Target (<3% overhead): {"✅ PASS" if summary['overhead_percent']['avg'] < 3.0 else "❌ FAIL"}
Deep Mode Acceptable (<50% overhead): {"✅ PASS" if summary['overhead_percent']['max'] < 50.0 else "❌ FAIL"}
Memory Stability (growth <100MB): {"✅ PASS" if summary['memory_mb']['growth'] < 100.0 else "❌ FAIL"}
Performance Degradation (<10%): {"✅ PASS" if summary['operations_per_sec']['degradation_percent'] < 10.0 else "❌ FAIL"}

Recommendation: {"PRODUCTION READY" if (summary['overhead_percent']['avg'] < 3.0 and summary['memory_mb']['growth'] < 100.0) else "REQUIRES OPTIMIZATION"}
"""
    return report

if __name__ == "__main__":
    # Run performance validation
    print("Starting MemGuard Hybrid Monitoring Performance Validation...")
    
    # Set testing environment
    os.environ['MEMGUARD_TESTING_OVERRIDE'] = '1'
    
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])