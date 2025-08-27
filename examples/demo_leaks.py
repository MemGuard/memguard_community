#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemGuard Demo Script - 60-Second Investor/Customer Demo

This script demonstrates all types of memory leaks MemGuard can detect and prevent.
Perfect for live demos, showing clear before/after results with actual cost savings.

Usage:
    python examples/demo_leaks.py --mode detect    # Safe detect-only mode
    python examples/demo_leaks.py --mode hybrid    # Show auto-cleanup for handles
    python examples/demo_leaks.py --quick          # 30-second demo
"""

import argparse
import asyncio
import time
import threading
import socket
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import memguard
from memguard.sampling import get_memory_tracker


class LeakDemoSuite:
    """Comprehensive leak demonstration suite for investor/customer demos."""
    
    def __init__(self, duration: int = 60, intensity: str = "medium"):
        self.duration = duration
        self.intensity = intensity
        self.leak_objects = {
            'files': [],
            'sockets': [],
            'tasks': [],
            'listeners': [],
            'caches': {}
        }
        self.demo_running = False
        self.stats = {
            'files_opened': 0,
            'sockets_created': 0,
            'tasks_spawned': 0,
            'listeners_added': 0,
            'cache_entries': 0
        }
    
    def print_banner(self, title: str):
        """Print formatted demo section banner."""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory and system stats."""
        tracker = get_memory_tracker()
        return {
            'memory_mb': tracker.get_rss_mb() if tracker.is_available() else 0.0,
            'timestamp': time.time()
        }
    
    def start_file_leaks(self):
        """Demonstrate file handle leaks - most common production issue."""
        self.print_banner("📂 FILE HANDLE LEAKS")
        print("Opening files without proper cleanup...")
        
        leak_count = {'medium': 50, 'high': 100, 'quick': 20}[self.intensity]
        
        for i in range(leak_count):
            try:
                # Create temporary files and "forget" to close them
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(f"Demo leak file {i}".encode())
                # Intentionally NOT calling temp_file.close()
                self.leak_objects['files'].append(temp_file)
                self.stats['files_opened'] += 1
                
                if i % 10 == 0:
                    print(f"  Opened {i+1} files (no close() calls)...")
                    
            except Exception as e:
                print(f"  File creation error: {e}")
        
        print(f"✅ Created {len(self.leak_objects['files'])} file handle leaks")
    
    def start_socket_leaks(self):
        """Demonstrate socket handle leaks - common in network services."""
        self.print_banner("🌐 SOCKET HANDLE LEAKS")
        print("Creating sockets and leaving them idle...")
        
        leak_count = {'medium': 30, 'high': 50, 'quick': 10}[self.intensity]
        
        for i in range(leak_count):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)  # Prevent hanging
                
                # Attempt to connect (will likely fail, but socket remains open)
                try:
                    sock.connect(('127.0.0.1', 65000 + (i % 100)))  # Non-existent ports
                except:
                    pass  # Expected to fail, leaving socket open
                
                self.leak_objects['sockets'].append(sock)
                self.stats['sockets_created'] += 1
                
                if i % 10 == 0:
                    print(f"  Created {i+1} idle sockets...")
                    
            except Exception as e:
                print(f"  Socket creation error: {e}")
        
        print(f"✅ Created {len(self.leak_objects['sockets'])} socket leaks")
    
    async def start_async_leaks(self):
        """Demonstrate asyncio task/timer leaks - modern Python issue."""
        self.print_banner("⚡ ASYNCIO TASK/TIMER LEAKS")
        print("Spawning asyncio tasks that never complete...")
        
        leak_count = {'medium': 100, 'high': 200, 'quick': 30}[self.intensity]
        
        async def leaky_task(task_id: int):
            """A task that runs indefinitely without proper cleanup."""
            while True:
                await asyncio.sleep(10)  # Simulate work
                # This task will never be canceled or awaited
        
        for i in range(leak_count):
            task = asyncio.create_task(leaky_task(i))
            self.leak_objects['tasks'].append(task)
            self.stats['tasks_spawned'] += 1
            
            if i % 25 == 0:
                print(f"  Spawned {i+1} runaway tasks...")
        
        print(f"✅ Created {len(self.leak_objects['tasks'])} asyncio task leaks")
    
    def start_cache_leaks(self):
        """Demonstrate unbounded cache growth - silent killer."""
        self.print_banner("💾 UNBOUNDED CACHE GROWTH")
        print("Growing caches without size limits...")
        
        # Create different types of growing caches
        caches = {
            'user_sessions': {},
            'request_cache': {},
            'computed_results': [],
            'event_log': []
        }
        
        growth_rate = {'medium': 1000, 'high': 2000, 'quick': 500}[self.intensity]
        
        for i in range(growth_rate):
            # Simulate session cache growth
            caches['user_sessions'][f'user_{i}'] = {
                'login_time': time.time(),
                'session_data': f'large_data_blob_{i}' * 100,  # 1KB+ per entry
                'preferences': {'theme': 'dark', 'lang': 'en', 'history': list(range(50))}
            }
            
            # Simulate request cache growth
            caches['request_cache'][f'request_{i}'] = {
                'url': f'/api/data/{i}',
                'response': {'data': list(range(100)), 'metadata': {'cached_at': time.time()}},
                'headers': {f'header_{j}': f'value_{j}' for j in range(20)}
            }
            
            # Simulate computed results growth
            caches['computed_results'].append({
                'computation_id': i,
                'result_matrix': [[j*k for j in range(50)] for k in range(50)],  # Large nested data
                'timestamp': time.time()
            })
            
            # Simulate event log growth
            caches['event_log'].append({
                'event_id': i,
                'event_type': f'user_action_{i % 10}',
                'payload': {'data': f'event_payload_{i}' * 50},
                'context': {'ip': f'192.168.1.{i % 255}', 'user_agent': 'Mozilla/5.0...' * 10}
            })
            
            self.stats['cache_entries'] += 4  # 4 different caches
            
            if i % 200 == 0:
                print(f"  Cache entries: {self.stats['cache_entries']:,}")
        
        self.leak_objects['caches'] = caches
        total_entries = sum(len(cache) if isinstance(cache, (dict, list)) else 1 for cache in caches.values())
        print(f"✅ Created {total_entries:,} cache entries across 4 collections")
    
    def start_cycle_leaks(self):
        """Demonstrate reference cycle leaks - subtle but costly."""
        self.print_banner("🔄 REFERENCE CYCLE LEAKS")
        print("Creating circular references...")
        
        class Node:
            def __init__(self, value):
                self.value = value
                self.refs = []
                self.parent = None
                self.metadata = {'created_at': time.time(), 'data': f'node_data_{value}' * 100}
        
        cycles = []
        cycle_count = {'medium': 50, 'high': 100, 'quick': 20}[self.intensity]
        
        for i in range(cycle_count):
            # Create a cycle of nodes
            nodes = [Node(f"{i}_{j}") for j in range(5)]
            
            # Create circular references
            for j, node in enumerate(nodes):
                node.refs.append(nodes[(j + 1) % len(nodes)])  # Point to next
                node.parent = nodes[(j - 1) % len(nodes)]       # Point to previous
            
            cycles.append(nodes)
            
            if i % 10 == 0:
                print(f"  Created {(i+1)*5} objects in {i+1} reference cycles...")
        
        self.leak_objects['cycles'] = cycles
        total_objects = sum(len(cycle) for cycle in cycles)
        print(f"✅ Created {total_objects} objects in {len(cycles)} reference cycles")
    
    async def run_leak_simulation(self):
        """Run the complete leak simulation."""
        print(f"\n🚀 Starting MemGuard Leak Demonstration")
        print(f"Duration: {self.duration}s | Intensity: {self.intensity}")
        print(f"This will create various types of memory leaks to demonstrate detection...")
        
        self.demo_running = True
        start_time = time.time()
        baseline_memory = self.get_memory_stats()
        
        print(f"\n📊 Baseline Memory: {baseline_memory['memory_mb']:.1f} MB")
        
        # Start different types of leaks
        self.start_file_leaks()
        await asyncio.sleep(1)
        
        self.start_socket_leaks()
        await asyncio.sleep(1)
        
        await self.start_async_leaks()
        await asyncio.sleep(1)
        
        self.start_cache_leaks()
        await asyncio.sleep(1)
        
        self.start_cycle_leaks()
        
        # Let leaks "mature" for detection
        remaining_time = self.duration - (time.time() - start_time)
        if remaining_time > 5:
            print(f"\n⏳ Letting leaks mature for {remaining_time:.0f} seconds...")
            await asyncio.sleep(remaining_time)
        
        current_memory = self.get_memory_stats()
        memory_growth = current_memory['memory_mb'] - baseline_memory['memory_mb']
        
        print(f"\n📈 Final Memory: {current_memory['memory_mb']:.1f} MB (+{memory_growth:.1f} MB growth)")
        print(f"\n📋 Leak Summary:")
        for leak_type, count in self.stats.items():
            print(f"  {leak_type}: {count:,}")
        
        self.demo_running = False
    
    def cleanup_demo_artifacts(self):
        """Clean up any demo artifacts (files, etc.)."""
        # Clean up temporary files
        for file_obj in self.leak_objects['files']:
            try:
                if hasattr(file_obj, 'name') and os.path.exists(file_obj.name):
                    os.unlink(file_obj.name)
            except:
                pass


async def run_memguard_demo(mode: str = "detect", duration: int = 60, intensity: str = "medium"):
    """
    Run the complete MemGuard demonstration.
    
    Args:
        mode: 'detect' (safe) or 'hybrid' (show auto-cleanup)
        duration: Demo duration in seconds
        intensity: 'quick', 'medium', or 'high' leak intensity
    """
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🛡️  MEMGUARD DEMO                                  ║
║                     Production Memory Leak Prevention                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 Demo Goal: Show real memory leaks being detected and prevented
🔧 Mode: {mode.upper()} | Duration: {duration}s | Intensity: {intensity.upper()}
💰 Expected savings: $200-2000/month per service
""")
    
    # Configure MemGuard based on mode
    if mode == "hybrid":
        # Enable auto-cleanup for handles only (safe demo)
        auto_cleanup = {
            'handles': True,     # Safe to auto-close files/sockets in demo
            'timers': True,      # Safe to cancel runaway tasks
            'listeners': False,  # Never auto-remove
            'cycles': False,     # Never auto-break 
            'caches': False      # Never auto-evict
        }
        print("🔧 MemGuard configured with HYBRID mode (auto-cleanup for handles/timers)")
    else:
        # Detect-only mode (production safe)
        auto_cleanup = {pattern: False for pattern in ['handles', 'timers', 'listeners', 'cycles', 'caches']}
        print("🔧 MemGuard configured with DETECT-ONLY mode (production safe)")
    
    # Start MemGuard protection
    print("\n🚀 Starting MemGuard protection...")
    memguard.protect(
        threshold_mb=10,  # Low threshold for demo
        poll_interval_s=5.0,
        sample_rate=0.1,  # Higher sampling for demo visibility
        auto_cleanup=auto_cleanup,
        aggressive_mode=True,  # More sensitive detection for demo
        background=True
    )
    
    print("✅ MemGuard protection active!")
    
    # Create and run leak simulation
    demo = LeakDemoSuite(duration=duration, intensity=intensity)
    
    try:
        await demo.run_leak_simulation()
        
        # Give MemGuard time to detect leaks
        print("\n🔍 Running MemGuard analysis...")
        await asyncio.sleep(2)
        
        # Get MemGuard report
        report = memguard.analyze(force_full_scan=True)
        
        # Display results
        print("\n" + "="*80)
        print("                         📊 MEMGUARD ANALYSIS RESULTS")
        print("="*80)
        
        if report.findings:
            print(f"\n🎯 LEAKS DETECTED: {len(report.findings)} findings")
            print(f"💰 ESTIMATED COST: ${report.total_estimated_cost_usd_per_month:.2f}/month")
            print(f"📈 MEMORY GROWTH: {report.memory_growth_mb:.1f} MB")
            print(f"⚡ SCAN TIME: {report.scan_duration_ms:.1f}ms")
            
            # Show top findings
            print(f"\n🏆 TOP FINDINGS:")
            for i, finding in enumerate(report.top_findings(5), 1):
                status_icon = "🔧" if mode == "hybrid" and finding.pattern in ['handles', 'timers'] else "🔍"
                print(f"  {i}. {status_icon} {finding.pattern.upper()}: {finding.location}")
                print(f"     Size: {finding.size_mb:.2f}MB | Confidence: {finding.confidence:.0%} | Severity: {finding.severity.value.upper()}")
                print(f"     Fix: {finding.suggested_fix}")
                print()
            
            # Show performance stats
            status = memguard.get_status()
            perf_stats = status.get('performance_stats', {})
            
            print(f"⚙️  PERFORMANCE METRICS:")
            print(f"   Overhead: {perf_stats.get('overhead_percentage', 0):.2f}%")
            print(f"   Scan Count: {status.get('scan_count', 0)}")
            print(f"   Avg Scan Time: {perf_stats.get('avg_scan_duration_ms', 0):.1f}ms")
            
            # Create investor summary
            print(f"\n💼 INVESTOR SUMMARY:")
            print(f"   MemGuard v1.0 — Overhead: {perf_stats.get('overhead_percentage', 0):.1f}% CPU")
            print(f"   Leaks detected: {len(report.findings)} | Est. savings: ${report.total_estimated_cost_usd_per_month:.0f}/month")
            print(f"   ROI: {report.total_estimated_cost_usd_per_month/100:.1f}x (vs $100/month MemGuard cost)")
            
        else:
            print("\n✅ No leaks detected (this would be unexpected in demo)")
            print("   This might indicate:")
            print("   - Demo intensity too low")
            print("   - MemGuard sampling too conservative")
            print("   - Need to wait longer for detection")
        
        # JSON export for enterprise reporting
        print(f"\n📄 JSON Report (for enterprise reporting):")
        print(json.dumps(report.to_dict(redact_sensitive=True), indent=2))
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up...")
        demo.cleanup_demo_artifacts()
        memguard.stop()
        print("✅ MemGuard protection stopped")
        print("🎬 Demo complete!")


def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(description="MemGuard Memory Leak Prevention Demo")
    parser.add_argument('--mode', choices=['detect', 'hybrid'], default='detect',
                       help='Demo mode: detect-only (safe) or hybrid (show auto-cleanup)')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Demo duration in seconds (default: 60)')
    parser.add_argument('--intensity', choices=['quick', 'medium', 'high'], default='medium',
                       help='Leak intensity level (default: medium)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick 30-second demo (equivalent to --duration 30 --intensity quick)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.duration = 30
        args.intensity = 'quick'
    
    # Run the demo
    try:
        asyncio.run(run_memguard_demo(
            mode=args.mode,
            duration=args.duration, 
            intensity=args.intensity
        ))
    except KeyboardInterrupt:
        print("\n👋 Demo terminated by user")
        sys.exit(0)


if __name__ == '__main__':
    main()