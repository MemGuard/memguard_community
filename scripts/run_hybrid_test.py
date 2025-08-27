#!/usr/bin/env python3
"""
Quick Test Runner for Hybrid Monitoring System
Runs a shorter validation test to demonstrate real metrics collection.
"""

import os
import sys
import time
from pathlib import Path

# Add memguard to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_30min_hybrid_monitoring import HybridMonitoringTest
import logging

def run_quick_test(duration_minutes: int = 10):
    """Run a quick test of the hybrid monitoring system."""
    
    print(f"🚀 Starting {duration_minutes}-Minute Hybrid Monitoring Test")
    print("=" * 60)
    
    # Set up environment
    os.environ['MEMGUARD_TESTING_OVERRIDE'] = '1'
    
    # Create test instance
    test = HybridMonitoringTest()
    
    try:
        # Measure baseline
        print("📏 Measuring baseline performance...")
        test.baseline_measurements = test.measure_baseline_performance()
        print(f"   Baseline: {test.baseline_measurements['operations_per_second']:.1f} ops/sec")
        
        # Start MemGuard
        print("🛡️  Starting MemGuard hybrid monitoring...")
        import memguard
        memguard.protect(
            monitoring_mode="hybrid",
            intensive_cleanup_schedule="never",
            license_key='MEMGUARD-PRO-001-GLOBAL'
        )
        print("   MemGuard started in hybrid mode")
        
        # Start workload
        print("⚙️  Starting workload simulation...")
        test.workload_simulator.start_continuous_workload()
        
        # Create some test leaks
        print("🔍 Creating test leak patterns...")
        # Create local file leaks for testing
        temp_files = []
        for i in range(15):
            temp_file = test.temp_dir / f"test_leak_{i}.tmp"
            with open(temp_file, 'w') as f:
                f.write(f"test leak data {i}" * 100)
            # Don't close some files to create leaks
            if i % 3 == 0:
                temp_files.append(open(temp_file, 'r'))
        
        print(f"   Created {len(temp_files)} file handle leaks")
        
        # Monitor for specified duration
        intervals = duration_minutes // 2  # 2-minute intervals for quick test
        
        print(f"📊 Monitoring for {duration_minutes} minutes ({intervals} intervals)...")
        
        for interval in range(1, intervals + 1):
            print(f"   Interval {interval}/{intervals} starting...")
            
            # Wait for interval (2 minutes for quick test)
            interval_start = time.time()
            while time.time() - interval_start < 120:  # 2 minutes
                time.sleep(10)
                elapsed = time.time() - interval_start
                if int(elapsed) % 30 == 0:  # Every 30 seconds
                    print(f"     {elapsed/120*100:.0f}% complete")
            
            # Collect metrics
            metrics = test.collect_interval_metrics(interval)
            test.interval_metrics.append(metrics)
            
            print(f"   Interval {interval} Results:")
            print(f"     Mode: {metrics.monitoring_mode} ({metrics.current_scan_type})")
            print(f"     Overhead: {metrics.overhead_percent:.2f}%")
            print(f"     Scans: {metrics.total_scans_performed}")
            print(f"     Leaks detected: {metrics.total_leaks_detected}")
            print(f"     Memory: {metrics.memory_current_mb:.1f}MB")
            
        # Generate report
        print("📋 Generating final report...")
        report = test.generate_comprehensive_report()
        
        # Display results
        print("\n" + "=" * 60)
        print("🎯 HYBRID MONITORING TEST RESULTS")
        print("=" * 60)
        print(f"Duration: {report.test_duration_minutes:.1f} minutes")
        print(f"Average Overhead: {report.average_overhead_percent:.2f}%")
        print(f"Peak Memory: {report.peak_memory_usage_mb:.1f}MB")
        print(f"Total Scans: {report.total_scans}")
        print(f"Light Scans: {report.light_scan_percentage:.1f}%")
        print(f"Deep Scans: {report.deep_scan_percentage:.1f}%")
        print(f"System Stability: {report.system_stability_score:.1f}%")
        print(f"Performance Assessment: {report.performance_assessment}")
        print(f"Recommended Mode: {report.recommended_production_mode}")
        print(f"Mode Effectiveness: {report.monitoring_mode_effectiveness}")
        
        # Assessment
        if report.average_overhead_percent < 5.0:
            status = "✅ EXCELLENT"
        elif report.average_overhead_percent < 15.0:
            status = "✅ GOOD"  
        elif report.average_overhead_percent < 30.0:
            status = "⚠️  ACCEPTABLE"
        else:
            status = "❌ NEEDS OPTIMIZATION"
            
        print(f"\nOverall Status: {status}")
        
        if report.performance_assessment in ["EXCELLENT", "GOOD"]:
            print("🚀 READY FOR PRODUCTION USE")
        else:
            print("🔧 Requires tuning for production")
            
        print(f"\n📁 Reports saved to: {test.reports_dir}")
        print("=" * 60)
        
        return report
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Cleanup
        test.workload_simulator.stop_workload()
        try:
            memguard.stop()
        except:
            pass
        
        # Close any open file handles
        for temp_file in temp_files:
            try:
                temp_file.close()
            except:
                pass
        
        try:
            import shutil
            shutil.rmtree(test.temp_dir)
        except:
            pass

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Hybrid Monitoring Test")
    parser.add_argument("--duration", type=int, default=6, 
                       help="Test duration in minutes (default: 6)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run test
    result = run_quick_test(args.duration)
    
    if result:
        print(f"\n✅ Test completed successfully!")
        if result.performance_assessment in ["EXCELLENT", "GOOD"]:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Needs optimization
    else:
        sys.exit(1)  # Failed

if __name__ == "__main__":
    main()