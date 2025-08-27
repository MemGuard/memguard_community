#=============================================================================
# File        : examples/website_performance_demo.py
# Project     : MemGuard v1.0
# Component   : Website Performance Demo - Real Website Impact Testing
# Description : Demonstrate MemGuard impact on actual web applications
#               • Flask app with MemGuard protection
#               • Performance comparison with/without protection
#               • Real HTTP request benchmarks
#               • Pro features demonstration
# Author      : Kyle Clouthier
# Version     : 1.0.0  
# Created     : 2025-08-27
#=============================================================================

import os
import sys
import time
import json
import threading
import statistics
from pathlib import Path

# Add memguard to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import memguard
from memguard import protect, stop

def create_simple_web_app():
    """Create a simple Flask web application for testing."""
    try:
        from flask import Flask, jsonify, request, render_template_string
        
        app = Flask(__name__)
        
        # Simple HTML template for testing
        HTML_TEMPLATE = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MemGuard Performance Demo</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .status { padding: 20px; border-radius: 5px; margin: 20px 0; }
                .protected { background-color: #d4edda; border: 1px solid #c3e6cb; }
                .unprotected { background-color: #f8d7da; border: 1px solid #f5c6cb; }
                .metrics { background-color: #e2e3e5; border: 1px solid #d6d8db; }
            </style>
        </head>
        <body>
            <h1>🛡️ MemGuard Website Protection Demo</h1>
            <div class="status protected">
                <h3>✅ Protection Status: {{ status }}</h3>
                <p>License: {{ license_type }}</p>
                <p>Mode: {{ monitoring_mode }}</p>
            </div>
            <div class="metrics">
                <h3>📊 Live Metrics</h3>
                <p>Overhead: {{ overhead }}%</p>
                <p>Memory: {{ memory_mb }} MB</p>
                <p>Leaks Detected: {{ leaks_detected }}</p>
                <p>Response Time: {{ response_time }}ms</p>
            </div>
            <p><a href="/api/test">Test API Endpoint</a> | <a href="/simulate-load">Simulate Load</a> | <a href="/">Refresh</a></p>
        </body>
        </html>
        """
        
        @app.route('/')
        def home():
            # Get MemGuard status
            try:
                from memguard.core import get_status, is_protecting
                from memguard.licensing import get_license_manager
                
                if is_protecting():
                    status_info = get_status()
                    license_manager = get_license_manager()
                    license_status = license_manager.get_license_status()
                    
                    status = "ACTIVE"
                    license_type = license_status['license_type']
                    monitoring_mode = status_info['configuration'].get('monitoring_mode', 'unknown') if 'configuration' in status_info else 'unknown'
                    overhead = status_info['performance_stats']['overhead_percentage']
                    memory_mb = status_info['performance_stats']['memory_current_mb']
                    leaks_detected = status_info['performance_stats']['total_findings']
                else:
                    status = "INACTIVE"
                    license_type = "None"
                    monitoring_mode = "None"
                    overhead = 0.0
                    memory_mb = 0.0
                    leaks_detected = 0
            except:
                status = "ERROR"
                license_type = "Unknown"
                monitoring_mode = "Unknown"
                overhead = 0.0
                memory_mb = 0.0
                leaks_detected = 0
            
            # Measure response time
            start = time.perf_counter()
            response_time = (time.perf_counter() - start) * 1000
            
            return render_template_string(HTML_TEMPLATE, 
                status=status,
                license_type=license_type,
                monitoring_mode=monitoring_mode,
                overhead=f"{overhead:.3f}",
                memory_mb=f"{memory_mb:.1f}",
                leaks_detected=leaks_detected,
                response_time=f"{response_time:.2f}"
            )
        
        @app.route('/api/test')
        def api_test():
            """Test API endpoint with some file operations."""
            import tempfile
            
            # Simulate typical web app operations
            operations = []
            start_time = time.perf_counter()
            
            # File operations (common in web apps)
            with tempfile.NamedTemporaryFile(mode='w', delete=True) as f:
                f.write("test data for api call")
                f.flush()
                operations.append("file_write")
            
            # Cache simulation
            cache = {f"key_{i}": f"value_{i}" for i in range(100)}
            operations.append("cache_create")
            
            # JSON response
            response_time = (time.perf_counter() - start_time) * 1000
            
            return jsonify({
                "status": "success",
                "operations": operations,
                "response_time_ms": round(response_time, 2),
                "memguard_protected": memguard.is_protecting(),
                "timestamp": time.time()
            })
        
        @app.route('/simulate-load')
        def simulate_load():
            """Simulate heavier load with potential memory leaks."""
            import tempfile
            import socket
            
            operations_count = 0
            start_time = time.perf_counter()
            
            # Create some temporary files (potential leaks)
            temp_files = []
            for i in range(20):
                try:
                    f = tempfile.NamedTemporaryFile(delete=False)
                    f.write(b"load simulation data" * 100)
                    temp_files.append(f.name)
                    f.close()
                    operations_count += 1
                except:
                    pass
            
            # Create some socket connections (potential leaks)
            sockets = []
            for i in range(10):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.1)
                    sockets.append(sock)
                    operations_count += 1
                except:
                    pass
            
            # Cache operations
            caches = []
            for i in range(5):
                cache = {f"heavy_key_{j}": "data" * 200 for j in range(100)}
                caches.append(cache)
                operations_count += 1
            
            # Clean up (simulate good cleanup)
            cleaned_up = 0
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                    cleaned_up += 1
                except:
                    pass
            
            for sock in sockets:
                try:
                    sock.close()
                    cleaned_up += 1
                except:
                    pass
            
            response_time = (time.perf_counter() - start_time) * 1000
            
            return jsonify({
                "status": "load_simulation_complete",
                "operations_performed": operations_count,
                "resources_cleaned": cleaned_up,
                "response_time_ms": round(response_time, 2),
                "memguard_active": memguard.is_protecting()
            })
        
        return app
        
    except ImportError:
        print("Flask not available. Install with: pip install flask")
        return None

def benchmark_website_performance():
    """Benchmark website performance with and without MemGuard."""
    try:
        import requests
    except ImportError:
        print("Requests library not available. Install with: pip install requests")
        return
    
    print("🌐 Website Performance Benchmark")
    print("=" * 50)
    
    # Test URLs
    base_url = "http://127.0.0.1:5000"
    test_endpoints = [
        "/",
        "/api/test",
        "/simulate-load"
    ]
    
    results = {
        "without_memguard": {},
        "with_memguard": {}
    }
    
    # Test without MemGuard
    print("\n1️⃣ Testing WITHOUT MemGuard protection...")
    stop()  # Ensure MemGuard is off
    
    for endpoint in test_endpoints:
        times = []
        for _ in range(10):  # 10 requests per endpoint
            try:
                start = time.perf_counter()
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
                end = time.perf_counter()
                
                if response.status_code == 200:
                    times.append((end - start) * 1000)  # Convert to ms
            except:
                pass
        
        if times:
            results["without_memguard"][endpoint] = {
                "avg_ms": statistics.mean(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "requests": len(times)
            }
    
    # Test with MemGuard
    print("2️⃣ Testing WITH MemGuard protection...")
    protect(
        monitoring_mode="hybrid",
        license_key="MEMGUARD-PRO-001-GLOBAL"
    )
    
    time.sleep(2)  # Allow MemGuard to initialize
    
    for endpoint in test_endpoints:
        times = []
        for _ in range(10):  # 10 requests per endpoint
            try:
                start = time.perf_counter()
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
                end = time.perf_counter()
                
                if response.status_code == 200:
                    times.append((end - start) * 1000)  # Convert to ms
            except:
                pass
        
        if times:
            results["with_memguard"][endpoint] = {
                "avg_ms": statistics.mean(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "requests": len(times)
            }
    
    # Calculate and display results
    print("\n📊 PERFORMANCE COMPARISON RESULTS")
    print("=" * 50)
    
    for endpoint in test_endpoints:
        without = results["without_memguard"].get(endpoint, {})
        with_mg = results["with_memguard"].get(endpoint, {})
        
        if without and with_mg:
            overhead_percent = ((with_mg["avg_ms"] - without["avg_ms"]) / without["avg_ms"]) * 100
            
            print(f"\n🔗 {endpoint}")
            print(f"   Without MemGuard: {without['avg_ms']:.2f}ms avg")
            print(f"   With MemGuard:    {with_mg['avg_ms']:.2f}ms avg")
            print(f"   Overhead:         {overhead_percent:+.2f}%")
            
            if overhead_percent < 3.0:
                print("   Status:           ✅ EXCELLENT (<3% overhead)")
            elif overhead_percent < 10.0:
                print("   Status:           ✅ ACCEPTABLE (<10% overhead)")
            else:
                print("   Status:           ⚠️  HIGH OVERHEAD")
    
    stop()  # Clean shutdown
    
    return results

def main():
    """Main demo interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MemGuard Website Performance Demo")
    parser.add_argument("--mode", choices=["server", "benchmark"], default="server",
                       help="Run web server or benchmark mode")
    parser.add_argument("--port", type=int, default=5000,
                       help="Web server port")
    
    args = parser.parse_args()
    
    if args.mode == "server":
        print("🚀 Starting MemGuard Website Demo Server...")
        
        # Create Flask app
        app = create_simple_web_app()
        if not app:
            return
        
        # Start MemGuard protection
        protect(
            monitoring_mode="hybrid",
            license_key="MEMGUARD-PRO-001-GLOBAL"
        )
        
        print(f"🌐 Demo website running at: http://127.0.0.1:{args.port}")
        print("   View real-time MemGuard metrics in your browser")
        print("   Press Ctrl+C to stop")
        
        try:
            app.run(host='127.0.0.1', port=args.port, debug=False)
        except KeyboardInterrupt:
            pass
        finally:
            stop()
    
    elif args.mode == "benchmark":
        # Start web server in background for benchmarking
        app = create_simple_web_app()
        if not app:
            return
        
        def run_server():
            app.run(host='127.0.0.1', port=args.port, debug=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        time.sleep(3)  # Wait for server to start
        
        try:
            benchmark_website_performance()
        except Exception as e:
            print(f"Benchmark error: {e}")

if __name__ == "__main__":
    # Set testing environment
    os.environ['MEMGUARD_TESTING_OVERRIDE'] = '1'
    main()