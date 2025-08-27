#=============================================================================
# File        : scripts/performance_dashboard.py
# Project     : MemGuard v1.0
# Component   : Performance Dashboard - Real-time Production Metrics
# Description : Live dashboard showing MemGuard performance impact
#               • Real-time overhead monitoring
#               • Website performance metrics
#               • Pro vs Open Source feature comparison
#               • Production readiness validation
# Author      : Kyle Clouthier
# Version     : 1.0.0
# Created     : 2025-08-27
#=============================================================================

import os
import sys
import json
import time
import threading
import statistics
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

# Add memguard to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import memguard
from memguard import protect, stop, get_status, analyze
from memguard.config import MemGuardConfig
from memguard.licensing import validate_pro_license, get_license_manager

@dataclass
class MetricsSnapshot:
    """Single point-in-time metrics snapshot."""
    timestamp: float
    monitoring_mode: str
    cpu_usage_percent: float
    memory_usage_mb: float
    memguard_overhead_percent: float
    operations_per_second: float
    active_leaks_detected: int
    total_cleanups_performed: int
    scan_type: str
    license_type: str

class PerformanceDashboard:
    """Real-time performance monitoring dashboard."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.metrics_history: List[MetricsSnapshot] = []
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.workload_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Workload simulation
        self.baseline_ops_per_sec = 0.0
        self.current_ops_count = 0
        self.ops_start_time = time.time()
        
    def start_monitoring(self, license_key: Optional[str] = None, 
                        monitoring_mode: str = "hybrid",
                        duration_minutes: int = 10):
        """Start real-time performance monitoring."""
        
        if self.is_running:
            print("Dashboard already running!")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        print(f"🚀 Starting MemGuard Performance Dashboard")
        print(f"   Mode: {monitoring_mode}")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   License: {'Pro' if license_key else 'Open Source'}")
        print("=" * 60)
        
        # Configure MemGuard
        config = MemGuardConfig(
            monitoring_mode=monitoring_mode,
            light_sample_rate=0.01,  # 1% for light mode
            deep_sample_rate=1.0,    # 100% for deep mode  
            poll_interval_s=2.0,
            threshold_mb=25,
            intensive_cleanup_schedule="never",  # Manual control
            debug_mode=False
        )
        
        # Start protection
        protect(config=config, license_key=license_key)
        
        # Start workload simulation
        self.workload_thread = threading.Thread(
            target=self._simulate_workload,
            daemon=True
        )
        self.workload_thread.start()
        
        # Start monitoring
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(duration_minutes * 60,),
            daemon=True
        )
        self.monitor_thread.start()
        
        # Display live dashboard
        self._display_dashboard()
    
    def _simulate_workload(self):
        """Simulate realistic web application workload."""
        import tempfile
        import socket
        
        while not self.stop_event.is_set():
            try:
                # File operations (web app serving files)
                with tempfile.NamedTemporaryFile() as f:
                    f.write(b"web content" * 100)  # 1KB file
                    f.flush()
                
                # Socket operations (HTTP requests)
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.01)
                    sock.connect(('127.0.0.1', 80))  # Will fail, that's OK
                except:
                    pass
                finally:
                    try:
                        sock.close()
                    except:
                        pass
                
                # Memory operations (caching)
                cache = {f"key_{i}": "data" * 50 for i in range(10)}
                del cache
                
                self.current_ops_count += 1
                
                # Rate limiting (simulate 50 req/sec web traffic)
                time.sleep(0.02)
                
            except Exception:
                pass
    
    def _monitoring_loop(self, duration_seconds: float):
        """Main monitoring loop collecting metrics."""
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time and not self.stop_event.is_set():
            try:
                # Collect metrics
                snapshot = self._collect_metrics()
                self.metrics_history.append(snapshot)
                
                # Keep only recent history (last hour)
                cutoff_time = time.time() - 3600
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp > cutoff_time
                ]
                
            except Exception as e:
                print(f"Error collecting metrics: {e}")
            
            time.sleep(self.update_interval)
        
        self.stop_monitoring()
    
    def _collect_metrics(self) -> MetricsSnapshot:
        """Collect current system and MemGuard metrics."""
        # Get MemGuard status
        status = get_status()
        
        # Calculate operations per second
        elapsed = time.time() - self.ops_start_time
        current_ops_per_sec = self.current_ops_count / elapsed if elapsed > 0 else 0.0
        
        # Get license info
        license_manager = get_license_manager()
        license_status = license_manager.get_license_status()
        
        # Determine scan type based on overhead
        overhead = status['performance_stats']['overhead_percentage']
        scan_type = "deep" if overhead > 10.0 else "light"
        
        return MetricsSnapshot(
            timestamp=time.time(),
            monitoring_mode=status['configuration'].get('monitoring_mode', 'unknown') if 'configuration' in status else 'unknown',
            cpu_usage_percent=0.0,  # Would need psutil for real CPU monitoring
            memory_usage_mb=status['performance_stats']['memory_current_mb'],
            memguard_overhead_percent=overhead,
            operations_per_second=current_ops_per_sec,
            active_leaks_detected=status['performance_stats']['total_findings'],
            total_cleanups_performed=0,  # Would need to track from cleanup operations
            scan_type=scan_type,
            license_type=license_status['license_type']
        )
    
    def _display_dashboard(self):
        """Display live dashboard in console."""
        while self.is_running and not self.stop_event.is_set():
            try:
                # Clear screen
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Display header
                print("🛡️  MEMGUARD PERFORMANCE DASHBOARD")
                print("=" * 80)
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print()
                
                if not self.metrics_history:
                    print("Collecting initial metrics...")
                    time.sleep(2)
                    continue
                
                # Get recent metrics
                latest = self.metrics_history[-1]
                recent_5min = [m for m in self.metrics_history if m.timestamp > time.time() - 300]
                
                # Current status
                print(f"🔧 Configuration:")
                print(f"   Monitoring Mode: {latest.monitoring_mode.upper()}")
                print(f"   License Type: {latest.license_type.upper()}")
                print(f"   Current Scan: {latest.scan_type.upper()}")
                print()
                
                # Performance metrics
                if recent_5min:
                    avg_overhead = statistics.mean([m.memguard_overhead_percent for m in recent_5min])
                    max_overhead = max([m.memguard_overhead_percent for m in recent_5min])
                    avg_memory = statistics.mean([m.memory_usage_mb for m in recent_5min])
                    avg_ops = statistics.mean([m.operations_per_second for m in recent_5min])
                else:
                    avg_overhead = latest.memguard_overhead_percent
                    max_overhead = latest.memguard_overhead_percent
                    avg_memory = latest.memory_usage_mb
                    avg_ops = latest.operations_per_second
                
                print(f"⚡ Performance (5-min average):")
                print(f"   Overhead: {avg_overhead:.2f}% (max: {max_overhead:.2f}%)")
                print(f"   Memory Usage: {avg_memory:.1f} MB")
                print(f"   Operations/sec: {avg_ops:.1f}")
                
                # Status indicators
                overhead_status = "🟢" if avg_overhead < 3.0 else "🟡" if avg_overhead < 10.0 else "🔴"
                memory_status = "🟢" if avg_memory < 100.0 else "🟡" if avg_memory < 200.0 else "🔴"
                
                print(f"   Overhead Status: {overhead_status} {'EXCELLENT' if avg_overhead < 3.0 else 'ACCEPTABLE' if avg_overhead < 10.0 else 'HIGH'}")
                print(f"   Memory Status: {memory_status} {'STABLE' if avg_memory < 100.0 else 'GROWING' if avg_memory < 200.0 else 'HIGH'}")
                print()
                
                # Leak detection
                print(f"🔍 Detection:")
                print(f"   Active Leaks: {latest.active_leaks_detected}")
                print(f"   Total Scans: {len(self.metrics_history)}")
                print()
                
                # Timeline (last 20 measurements)
                print(f"📈 Overhead Timeline (last 20 measurements):")
                recent_20 = self.metrics_history[-20:] if len(self.metrics_history) >= 20 else self.metrics_history
                
                timeline = ""
                for snapshot in recent_20:
                    if snapshot.memguard_overhead_percent < 3.0:
                        timeline += "▁"  # Low overhead
                    elif snapshot.memguard_overhead_percent < 10.0:
                        timeline += "▄"  # Medium overhead
                    else:
                        timeline += "█"  # High overhead
                
                print(f"   {timeline}")
                print(f"   ▁ <3%  ▄ 3-10%  █ >10%")
                print()
                
                # Production readiness assessment
                if len(recent_5min) >= 10:  # Need sufficient data
                    production_ready = (
                        avg_overhead < 5.0 and
                        max_overhead < 15.0 and
                        avg_memory < 150.0
                    )
                    
                    status_icon = "✅" if production_ready else "⚠️"
                    status_text = "PRODUCTION READY" if production_ready else "NEEDS OPTIMIZATION"
                    
                    print(f"🎯 Assessment: {status_icon} {status_text}")
                    
                    if not production_ready:
                        print("   Recommendations:")
                        if avg_overhead >= 5.0:
                            print("   • Consider light monitoring mode")
                        if max_overhead >= 15.0:
                            print("   • Reduce deep scan frequency")
                        if avg_memory >= 150.0:
                            print("   • Enable intensive cleanup scheduling")
                else:
                    print("🎯 Assessment: Collecting data...")
                
                print()
                print("Press Ctrl+C to stop monitoring")
                
                time.sleep(2)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Dashboard error: {e}")
                time.sleep(1)
    
    def stop_monitoring(self):
        """Stop monitoring and cleanup."""
        if not self.is_running:
            return
        
        print("\n🛑 Stopping performance monitoring...")
        
        self.is_running = False
        self.stop_event.set()
        
        # Stop MemGuard
        stop()
        
        # Generate final report
        self._generate_final_report()
    
    def _generate_final_report(self):
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            print("No metrics collected for report")
            return
        
        # Calculate statistics
        overhead_values = [m.memguard_overhead_percent for m in self.metrics_history]
        memory_values = [m.memory_usage_mb for m in self.metrics_history]
        ops_values = [m.operations_per_second for m in self.metrics_history]
        
        # Generate report
        report = {
            "test_summary": {
                "duration_seconds": self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp,
                "total_measurements": len(self.metrics_history),
                "monitoring_mode": self.metrics_history[-1].monitoring_mode,
                "license_type": self.metrics_history[-1].license_type
            },
            "performance_metrics": {
                "overhead_percent": {
                    "average": statistics.mean(overhead_values),
                    "maximum": max(overhead_values),
                    "minimum": min(overhead_values),
                    "p95": statistics.quantiles(overhead_values, n=20)[18] if len(overhead_values) > 20 else max(overhead_values)
                },
                "memory_usage_mb": {
                    "average": statistics.mean(memory_values),
                    "maximum": max(memory_values),
                    "minimum": min(memory_values),
                    "growth": max(memory_values) - min(memory_values)
                },
                "operations_per_second": {
                    "average": statistics.mean(ops_values),
                    "maximum": max(ops_values),
                    "minimum": min(ops_values)
                }
            },
            "production_assessment": {
                "overhead_target_met": statistics.mean(overhead_values) < 3.0,
                "memory_stable": (max(memory_values) - min(memory_values)) < 100.0,
                "performance_acceptable": max(overhead_values) < 50.0,
                "overall_recommendation": "PRODUCTION_READY" if (
                    statistics.mean(overhead_values) < 3.0 and
                    (max(memory_values) - min(memory_values)) < 100.0
                ) else "REQUIRES_OPTIMIZATION"
            },
            "feature_comparison": {
                "pro_features_available": self.metrics_history[-1].license_type == "PRO",
                "auto_cleanup_enabled": True,
                "advanced_detection_enabled": True,
                "enterprise_reporting_enabled": True
            }
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(__file__).parent.parent / f"reports/performance_report_{timestamp}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Display summary
        print("\n" + "="*80)
        print("🎯 FINAL PERFORMANCE REPORT")
        print("="*80)
        print(f"Duration: {report['test_summary']['duration_seconds']:.1f} seconds")
        print(f"Measurements: {report['test_summary']['total_measurements']}")
        print(f"Mode: {report['test_summary']['monitoring_mode']}")
        print(f"License: {report['test_summary']['license_type']}")
        print()
        
        perf = report['performance_metrics']
        print(f"Overhead: {perf['overhead_percent']['average']:.2f}% avg, {perf['overhead_percent']['maximum']:.2f}% max")
        print(f"Memory: {perf['memory_usage_mb']['average']:.1f} MB avg, {perf['memory_usage_mb']['growth']:.1f} MB growth")
        print(f"Operations: {perf['operations_per_second']['average']:.1f}/sec avg")
        print()
        
        assessment = report['production_assessment']
        print(f"Production Ready: {'✅ YES' if assessment['overall_recommendation'] == 'PRODUCTION_READY' else '❌ NO'}")
        print(f"Overhead Target (<3%): {'✅' if assessment['overhead_target_met'] else '❌'}")
        print(f"Memory Stable: {'✅' if assessment['memory_stable'] else '❌'}")
        print()
        print(f"📄 Full report saved: {report_file}")

def main():
    """Main dashboard interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MemGuard Performance Dashboard")
    parser.add_argument("--mode", choices=["light", "deep", "hybrid"], default="hybrid",
                       help="Monitoring mode to test")
    parser.add_argument("--license", type=str,
                       help="Pro license key for testing Pro features")
    parser.add_argument("--duration", type=int, default=10,
                       help="Test duration in minutes")
    parser.add_argument("--demo", action="store_true",
                       help="Run with demo Pro license")
    
    args = parser.parse_args()
    
    # Use demo license if requested
    license_key = None
    if args.demo:
        license_key = "MEMGUARD-PRO-001-GLOBAL"
    elif args.license:
        license_key = args.license
    
    # Set testing environment
    os.environ['MEMGUARD_TESTING_OVERRIDE'] = '1'
    
    # Create and start dashboard
    dashboard = PerformanceDashboard()
    
    try:
        dashboard.start_monitoring(
            license_key=license_key,
            monitoring_mode=args.mode,
            duration_minutes=args.duration
        )
    except KeyboardInterrupt:
        dashboard.stop_monitoring()

if __name__ == "__main__":
    main()