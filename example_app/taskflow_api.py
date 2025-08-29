#!/usr/bin/env python3
"""
TaskFlow - A Real-World Task Management API
==========================================
This is a realistic Python application that demonstrates common patterns that can lead to memory leaks.
A developer would typically build something like this for a startup or small business.

Features:
- REST API for task management
- File upload capabilities  
- In-memory caching (potential leak source)
- Database operations
- Background processing simulation

Common leak patterns this app exhibits:
1. Unclosed database connections
2. Growing caches without eviction  
3. File handles not properly closed
4. Session data growing unbounded
5. Background task accumulation

MemGuard Integration:
- Monitors the app in real-time
- Auto-cleanup enabled for production safety
- Reports available via API endpoints
"""

import asyncio
import os
import json
import time
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path
import tempfile
import shutil

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ======================================
# MEMGUARD INTEGRATION - Real Production Setup
# ======================================
print("🔧 Initializing MemGuard for TaskFlow API...")

try:
    import memguard
    # Using pure engine defaults for authentic production testing
    memguard.protect(
        # Open source defaults - all features available:
        # threshold_mb=10, poll_interval_s=1.0, sample_rate=0.01
        # patterns=('handles', 'caches', 'timers', 'cycles', 'listeners')
        # Auto-cleanup available with pattern configuration
        auto_cleanup={
            'handles': True,    # Auto-close abandoned files/sockets
            'caches': True,     # Auto-evict growing caches
            'timers': True,     # Auto-cancel orphaned timers
        }
    )
    print("✅ MemGuard open source monitoring ENABLED with FULL FEATURES!")
    print("🛡️  Mode: Default | Threshold: 10MB | Polling: 1s | Sampling: 1%")
    print("🧹 Auto-cleanup: ENABLED for handles, caches, timers")
    print("📊 Telemetry: Local metrics collection only")
    
except ImportError:
    print("❌ MemGuard not available - install MemGuard to enable monitoring")
    print("   Run: pip install memguard")

# ======================================
# APP SETUP
# ======================================
app = FastAPI(
    title="TaskFlow API",
    description="Real-world task management with MemGuard monitoring",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================
# GLOBAL STATE (Simulates Real App Data Structures)  
# ======================================

# App startup time for health check uptime calculation
app_start_time = time.time()
# These represent realistic data structures a developer would create

# Task cache - grows as tasks are accessed
task_cache = {}

# User sessions - accumulates as users interact
user_sessions = {}

# File processing cache - grows with uploads
file_processing_cache = {}

# Analytics data - collects metrics over time
analytics_events = []

# Background job queue - simulates async processing
background_jobs = []

# Database connection pool simulation
db_connections = []

# ======================================
# DATABASE SETUP (Simple SQLite)
# ======================================
DB_PATH = "taskflow.db"

def init_database():
    """Initialize the database with tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            priority INTEGER DEFAULT 1
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("📊 Database initialized")

# ======================================
# MEMORY LEAK SIMULATION FUNCTIONS
# ======================================

def simulate_realistic_memory_patterns():
    """
    Simulate memory usage patterns that real applications exhibit.
    These are NOT intentional leaks but common patterns that can accumulate.
    """
    timestamp = time.time()
    
    # 1. Cache data that accumulates over time (realistic pattern)
    cache_key = f"data_{int(timestamp)}"
    task_cache[cache_key] = {
        'data': f"cached_data_{'x' * 1000}",  # 1KB typical cache entry
        'created_at': timestamp,
        'access_count': 0,
        'metadata': {'source': 'api', 'version': '1.0'}
    }
    
    # 2. Session tracking (realistic for web apps)
    session_id = f"session_{len(user_sessions)}"
    user_sessions[session_id] = {
        'created_at': datetime.utcnow(),
        'last_access': datetime.utcnow(),
        'user_data': {'preferences': 'y' * 500},  # 500B user data
        'activity_log': []
    }
    
    # 3. Analytics/metrics collection (realistic for monitoring)
    analytics_events.append({
        'timestamp': datetime.utcnow(),
        'event_type': 'api_request',
        'endpoint': '/tasks',
        'response_time_ms': 45,
        'payload_size': len('z' * 200)  # 200B typical event
    })
    
    # 4. Background job simulation (realistic async processing)
    background_jobs.append({
        'job_id': len(background_jobs),
        'type': 'data_processing',
        'status': 'queued',
        'created_at': timestamp,
        'payload': 'w' * 800  # 800B job data
    })
    
    # Only keep recent data (simulate basic cleanup attempts)
    # This shows realistic app behavior where developers try to manage memory
    # but might not get the thresholds right
    
    # More aggressive cleanup for long-running tests
    if len(task_cache) > 500:  # Reduced from 1000 for stability
        # Handle different cache entry structures (some use 'created_at', others 'timestamp')
        try:
            oldest_key = min(task_cache.keys(), key=lambda k: task_cache[k].get('created_at', task_cache[k].get('timestamp', 0)))
            del task_cache[oldest_key]
        except (KeyError, ValueError):
            # Fallback: just remove first entry if structure parsing fails
            if task_cache:
                first_key = next(iter(task_cache))
                del task_cache[first_key]
    
    # Cleanup user sessions - remove sessions older than 1 hour
    if len(user_sessions) > 100:  # Prevent unbounded growth
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        old_sessions = [k for k, v in user_sessions.items() 
                       if v['created_at'] < cutoff_time]
        for session_id in old_sessions[:50]:  # Remove up to 50 old sessions
            del user_sessions[session_id]
    
    # Cleanup background jobs - remove completed/old jobs
    if len(background_jobs) > 200:  # Prevent job queue growth
        # Remove oldest jobs
        background_jobs[:] = background_jobs[-100:]  # Keep last 100
    
    if len(analytics_events) > 2000:  # Reduced from 5000 for stability
        analytics_events[:] = analytics_events[-1000:]  # Keep last 1000 events

def get_database_connection():
    """Get database connection - simulates connection management"""
    # This simulates how developers often handle DB connections
    # Sometimes they create new connections without proper pooling
    conn = sqlite3.connect(DB_PATH)
    db_connections.append({'conn': conn, 'created_at': time.time()})
    
    # More aggressive connection cleanup for stability
    if len(db_connections) > 5:  # Reduced from 10 for stability
        old_conn_info = db_connections.pop(0)
        try:
            old_conn_info['conn'].close()
        except:
            pass  # Connection might already be closed
    
    # Additional cleanup: close very old connections (>5 minutes)
    current_time = time.time()
    old_connections = []
    for i, conn_info in enumerate(db_connections):
        if current_time - conn_info['created_at'] > 300:  # 5 minutes
            old_connections.append(i)
    
    # Remove old connections (reverse order to maintain indices)
    for i in reversed(old_connections):
        old_conn_info = db_connections.pop(i)
        try:
            old_conn_info['conn'].close()
        except:
            pass
    
    return conn

# ======================================
# API ENDPOINTS
# ======================================

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and tests - RELOADED"""
    uptime = time.time() - app_start_time
    
    # Try to get MemGuard status
    memguard_healthy = False
    try:
        import memguard
        if memguard.is_protecting():
            memguard_healthy = True
    except:
        pass
    
    # Force periodic cleanup on health checks to maintain stability
    if uptime > 60:  # After 1 minute uptime
        simulate_realistic_memory_patterns()  # This will trigger cleanup
    
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "timestamp": datetime.utcnow().isoformat(),
        "memguard_active": memguard_healthy,
        "app_version": "2.0.1",  # Version bump for stability fixes
        "service": "TaskFlow API",
        "resource_usage": {
            "task_cache_size": len(task_cache),
            "user_sessions": len(user_sessions),
            "analytics_events": len(analytics_events),
            "background_jobs": len(background_jobs),
            "db_connections": len(db_connections)
        }
    }

@app.get("/")
async def root():
    """Root endpoint - shows app status and MemGuard integration"""
    simulate_realistic_memory_patterns()
    
    # Try to get MemGuard status
    memguard_status = "Not Available"
    try:
        import memguard
        if memguard.is_protecting():
            status = memguard.get_status()
            memguard_status = {
                "monitoring": True,
                "memory_mb": status.get('memory_current_mb', 0),
                "findings": len(status.get('latest_findings', []))
            }
        else:
            memguard_status = {"monitoring": False}
    except:
        pass
    
    return {
        "app": "TaskFlow API v2.0",
        "description": "Real-world task management with MemGuard monitoring",
        "status": "running",
        "memguard": memguard_status,
        "app_metrics": {
            "cached_tasks": len(task_cache),
            "active_sessions": len(user_sessions),
            "analytics_events": len(analytics_events),
            "background_jobs": len(background_jobs),
            "db_connections": len(db_connections)
        },
        "endpoints": {
            "tasks": "/tasks/ (GET, POST)",
            "upload": "/upload/ (POST)",
            "analytics": "/analytics/ (GET)",
            "memguard_status": "/memguard/status (GET)",
            "memguard_report": "/memguard/report (GET)",
            "stress_test": "/stress-test/ (POST)"
        }
    }

@app.get("/tasks/")
async def get_tasks():
    """Get all tasks with realistic caching"""
    simulate_realistic_memory_patterns()
    
    # Check cache first (realistic app pattern)
    cache_key = "all_tasks_list"
    cached_result = task_cache.get(cache_key)
    
    if cached_result and time.time() - cached_result['created_at'] < 30:
        cached_result['access_count'] += 1
        return {"tasks": cached_result['data'], "source": "cache"}
    
    # Get from database
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT 50")
        rows = cursor.fetchall()
        conn.close()
        
        tasks = [
            {
                "id": row[0],
                "title": row[1], 
                "description": row[2],
                "status": row[3],
                "created_at": row[4],
                "priority": row[5]
            }
            for row in rows
        ]
        
        # Cache the result
        task_cache[cache_key] = {
            'data': tasks,
            'created_at': time.time(),
            'access_count': 1
        }
        
        return {"tasks": tasks, "source": "database", "count": len(tasks)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/tasks/")
async def create_task(title: str, description: str = "", priority: int = 1):
    """Create a new task"""
    simulate_realistic_memory_patterns()
    
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO tasks (title, description, priority) VALUES (?, ?, ?)",
            (title, description, priority)
        )
        task_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Cache the new task
        task_cache[f"task_{task_id}"] = {
            'id': task_id,
            'title': title,
            'description': description,
            'priority': priority,
            'created_at': time.time()
        }
        
        # Invalidate list cache
        task_cache.pop("all_tasks_list", None)
        
        return {
            "id": task_id,
            "title": title,
            "status": "created",
            "message": "Task created successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """File upload with processing simulation"""
    simulate_realistic_memory_patterns()
    
    try:
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Create temporary file for processing
        temp_dir = tempfile.mkdtemp()
        temp_file_path = Path(temp_dir) / file.filename
        
        # Write file (potential handle leak if not managed properly)
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        # Simulate file processing
        processing_result = {
            'filename': file.filename,
            'size_bytes': file_size,
            'processed_at': time.time(),
            'temp_path': str(temp_file_path),
            'processing_time_ms': 150
        }
        
        # Cache processing result
        file_processing_cache[file.filename] = processing_result
        
        # Clean up temp file (good practice)
        try:
            temp_file_path.unlink()
            temp_dir.rmdir() if temp_dir.exists() else None
        except:
            pass  # Sometimes cleanup fails
        
        return {
            "filename": file.filename,
            "size_kb": round(file_size / 1024, 2),
            "status": "processed",
            "processed_files_cached": len(file_processing_cache)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/analytics/")
async def get_analytics():
    """Get app analytics"""
    simulate_realistic_memory_patterns()
    
    total_events = len(analytics_events)
    recent_events = [e for e in analytics_events if 
                    (datetime.utcnow() - e['timestamp']).seconds < 3600]
    
    return {
        "total_events": total_events,
        "recent_events_1h": len(recent_events),
        "avg_response_time_ms": 45.5,
        "active_sessions": len(user_sessions),
        "cache_hit_ratio": 0.85,
        "background_jobs_queued": len(background_jobs)
    }

@app.post("/stress-test/")
async def stress_test(iterations: int = 50):
    """Stress test to trigger memory usage and MemGuard monitoring"""
    print(f"🔥 Starting stress test with {iterations} iterations...")
    
    results = []
    start_time = time.time()
    
    for i in range(iterations):
        # Create realistic data that accumulates
        simulate_realistic_memory_patterns()
        
        # Simulate database operations
        try:
            conn = get_database_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tasks")
            count = cursor.fetchone()[0]
            conn.close()
        except:
            count = 0
        
        # Create some temporary objects
        temp_data = {
            'iteration': i,
            'large_data': 'x' * 2000,  # 2KB per iteration
            'nested_data': {
                'level1': 'y' * 500,
                'level2': {'data': 'z' * 300}
            },
            'timestamp': time.time()
        }
        
        # Add to various collections
        task_cache[f"stress_test_{i}"] = temp_data
        analytics_events.append({
            'timestamp': datetime.utcnow(),
            'event_type': 'stress_test',
            'iteration': i,
            'data': temp_data
        })
        
        results.append({'iteration': i, 'db_tasks': count})
        
        # Small delay to allow MemGuard monitoring and prevent overwhelming
        if i % 10 == 0:
            await asyncio.sleep(0.1)
        
        # Periodic cleanup during stress test to prevent resource exhaustion
        if i % 50 == 0:
            simulate_realistic_memory_patterns()  # Triggers cleanup
    
    duration = time.time() - start_time
    
    # Try to get MemGuard report after stress test
    memguard_info = {}
    try:
        import memguard
        if memguard.is_protecting():
            report = memguard.get_report()
            memguard_info = {
                "monitoring_active": True,
                "total_findings": len(report.findings),
                "memory_current_mb": report.memory_current_mb,
                "critical_findings": len(report.critical_findings)
            }
    except:
        memguard_info = {"monitoring_active": False}
    
    return {
        "stress_test_completed": True,
        "iterations": iterations,
        "duration_seconds": round(duration, 2),
        "final_metrics": {
            "cached_items": len(task_cache),
            "active_sessions": len(user_sessions),
            "analytics_events": len(analytics_events),
            "background_jobs": len(background_jobs),
            "db_connections": len(db_connections)
        },
        "memguard": memguard_info
    }

@app.get("/memguard/status")
async def memguard_status():
    """Get MemGuard monitoring status"""
    try:
        import memguard
        if memguard.is_protecting():
            status = memguard.get_status()
            return {
                "memguard_enabled": True,
                "monitoring_active": True,
                "status": status
            }
        else:
            return {
                "memguard_enabled": True,
                "monitoring_active": False,
                "message": "MemGuard not currently protecting"
            }
    except ImportError:
        return {
            "memguard_enabled": False,
            "message": "MemGuard not installed. Run: pip install memguard"
        }
    except Exception as e:
        return {
            "memguard_enabled": False,
            "error": str(e)
        }

@app.post("/admin/cleanup")
async def force_cleanup():
    """Force cleanup of application resources for stability"""
    initial_sizes = {
        "task_cache": len(task_cache),
        "user_sessions": len(user_sessions),
        "analytics_events": len(analytics_events),
        "background_jobs": len(background_jobs),
        "db_connections": len(db_connections)
    }
    
    # Aggressive cleanup
    task_cache.clear()
    user_sessions.clear()
    analytics_events.clear()
    background_jobs.clear()
    
    # Close all database connections
    for conn_info in db_connections[:]:
        try:
            conn_info['conn'].close()
        except:
            pass
    db_connections.clear()
    
    final_sizes = {
        "task_cache": len(task_cache),
        "user_sessions": len(user_sessions),
        "analytics_events": len(analytics_events),
        "background_jobs": len(background_jobs),
        "db_connections": len(db_connections)
    }
    
    return {
        "cleanup_performed": True,
        "initial_sizes": initial_sizes,
        "final_sizes": final_sizes,
        "cleanup_timestamp": datetime.utcnow().isoformat(),
        "message": "All application resources cleaned up"
    }

@app.get("/memguard/report")
async def memguard_report():
    """Get detailed MemGuard analysis report"""
    try:
        import memguard
        if memguard.is_protecting():
            report = memguard.get_report()
            
            # Get cleanup statistics if available
            cleanup_stats = {}
            try:
                from memguard.detectors.caches import get_global_cleanup_stats
                cleanup_stats['cache_cleanup'] = get_global_cleanup_stats()
            except:
                pass
            
            return {
                "memguard_enabled": True,
                "report": {
                    "scan_timestamp": datetime.utcnow().isoformat(),
                    "total_findings": len(report.findings),
                    "critical_findings": len(report.critical_findings),
                    "scan_duration_ms": report.scan_duration_ms,
                    "memory_baseline_mb": report.memory_baseline_mb,
                    "memory_current_mb": report.memory_current_mb,
                    "memory_growth_mb": report.memory_current_mb - report.memory_baseline_mb,
                    "cleanup_stats": cleanup_stats,
                    "top_findings": [
                        {
                            "pattern": f.pattern,
                            "location": f.location,
                            "size_mb": round(f.size_mb, 3),
                            "detail": f.detail,
                            "suggested_fix": f.suggested_fix,
                            "severity": f.severity.value if hasattr(f.severity, 'value') else str(f.severity)
                        }
                        for f in report.findings[:5]  # Top 5 findings
                    ]
                }
            }
        else:
            return {
                "memguard_enabled": True,
                "message": "MemGuard not currently active"
            }
    except ImportError:
        return {
            "memguard_enabled": False,
            "message": "MemGuard not installed"
        }
    except Exception as e:
        return {
            "memguard_enabled": False,
            "error": str(e)
        }


# ======================================
# STARTUP/SHUTDOWN
# ======================================

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    print("🚀 TaskFlow API starting up...")
    init_database()
    
    # Add some sample data
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO tasks (title, description) VALUES (?, ?)", 
                      ("Welcome Task", "This is a sample task to get you started"))
        cursor.execute("INSERT OR IGNORE INTO users (username, email) VALUES (?, ?)",
                      ("demo_user", "demo@taskflow.dev"))
        conn.commit()
        conn.close()
        print("📊 Sample data initialized")
    except:
        pass
    
    print("✅ TaskFlow API ready!")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🛡️  MemGuard Status: http://localhost:8000/memguard/status")

# =============================================================================
# LEAK GENERATION ENDPOINTS FOR COMPREHENSIVE AUTO-CLEANUP TESTING
# =============================================================================

# Global leak storage (deliberately not cleaned up to create genuine leaks)
_leaked_files = []
_leaked_sockets = []
_leaked_caches = {}
_leaked_tasks = []
_leaked_timers = []
_leaked_listeners = []

# GROUND-TRUTH LEAK TRACKING for accurate detection rate calculation
_ground_truth_leaks = {
    'files_injected': 0,
    'sockets_injected': 0, 
    'caches_injected': 0,
    'asyncio_tasks_injected': 0,
    'asyncio_timers_injected': 0,
    'total_injected': 0,
    'injection_log': []  # Detailed log for audit trail
}

@app.post("/test/create-file-leaks/{count}")
async def create_file_leaks(count: int):
    """Generate genuine file handle leaks for auto-cleanup testing"""
    import tempfile
    import threading
    
    leaked_files = []
    for i in range(count):
        # Create temp files that won't be closed (genuine leak)
        f = tempfile.NamedTemporaryFile(mode='w', delete=False, 
                                       prefix=f'app_leak_file_{threading.get_ident()}_')
        f.write(f"App leaked file {i} created at {datetime.utcnow()}")
        f.flush()
        # Store file handle but don't close - creates genuine leak
        _leaked_files.append(f)
        leaked_files.append(f.name)
    
    # GROUND-TRUTH TRACKING: Record injected leaks for detection rate calculation
    _ground_truth_leaks['files_injected'] += count
    _ground_truth_leaks['total_injected'] += count
    _ground_truth_leaks['injection_log'].append({
        'timestamp': time.time(),
        'type': 'file_handles',
        'count': count,
        'files': leaked_files
    })
    
    return {
        "leaked_files": leaked_files,
        "count": len(leaked_files),
        "total_leaked_files": len(_leaked_files),
        "ground_truth_injected": _ground_truth_leaks['files_injected']
    }

@app.post("/test/create-socket-leaks/{count}")
async def create_socket_leaks(count: int):
    """Generate genuine socket leaks for auto-cleanup testing"""
    import socket
    import threading
    
    leaked_sockets = []
    successful_sockets = 0
    for i in range(count):
        try:
            # Create socket that won't be closed (genuine leak)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('127.0.0.1', 0))  # Bind to random port
            sock.listen(1)
            port = sock.getsockname()[1]
            # Store socket but don't close - creates genuine leak
            _leaked_sockets.append(sock)
            leaked_sockets.append(f"127.0.0.1:{port}")
            successful_sockets += 1
        except Exception as e:
            print(f"Socket leak creation error: {e}")
    
    # GROUND-TRUTH TRACKING: Record injected leaks for detection rate calculation
    _ground_truth_leaks['sockets_injected'] += successful_sockets
    _ground_truth_leaks['total_injected'] += successful_sockets
    _ground_truth_leaks['injection_log'].append({
        'timestamp': time.time(),
        'type': 'socket_handles',
        'count': successful_sockets,
        'sockets': leaked_sockets
    })
    
    return {
        "leaked_sockets": leaked_sockets,
        "count": len(leaked_sockets),
        "total_leaked_sockets": len(_leaked_sockets),
        "ground_truth_injected": _ground_truth_leaks['sockets_injected']
    }

@app.post("/test/create-cache-leaks/{size_mb}")
async def create_cache_leaks(size_mb: int):
    """Generate genuine cache leaks for auto-cleanup testing"""
    import threading
    import random
    import string
    
    cache_key = f"app_leak_cache_{threading.get_ident()}_{datetime.utcnow().timestamp()}"
    
    # Generate large data that won't be cleaned up (genuine cache leak)
    data_size = size_mb * 1024 * 1024  # Convert MB to bytes
    leaked_data = []
    
    # Create chunks of random data
    chunk_size = 1024
    for _ in range(data_size // chunk_size):
        chunk = ''.join(random.choices(string.ascii_letters + string.digits, k=chunk_size))
        leaked_data.append(chunk)
    
    # Store in cache but never remove - creates genuine cache leak
    _leaked_caches[cache_key] = {
        "data": leaked_data,
        "created_at": datetime.utcnow(),
        "size_mb": size_mb,
        "chunks": len(leaked_data)
    }
    
    # GROUND-TRUTH TRACKING: Record injected cache leaks
    _ground_truth_leaks['caches_injected'] += 1
    _ground_truth_leaks['total_injected'] += 1
    _ground_truth_leaks['injection_log'].append({
        'timestamp': time.time(),
        'type': 'cache_leak',
        'count': 1,
        'size_mb': size_mb,
        'cache_key': cache_key
    })
    
    return {
        "cache_key": cache_key,
        "size_mb": size_mb,
        "chunks_created": len(leaked_data),
        "total_cache_keys": len(_leaked_caches),
        "total_cache_size_mb": sum(cache["size_mb"] for cache in _leaked_caches.values()),
        "ground_truth_injected": _ground_truth_leaks['caches_injected']
    }

@app.post("/test/create-asyncio-leaks/{count}")
async def create_asyncio_leaks(count: int):
    """Generate genuine asyncio task and timer leaks for auto-cleanup testing"""
    import asyncio
    import threading
    
    leaked_tasks = []
    leaked_timers = []
    
    async def leaked_task(task_id: int):
        """Task that runs forever without being awaited (genuine leak)"""
        try:
            while True:
                await asyncio.sleep(1)  # Infinite loop task
        except asyncio.CancelledError:
            pass
    
    def leaked_timer_callback(timer_id: int):
        """Timer callback that does nothing (genuine leak)"""
        pass
    
    # Create leaked tasks
    for i in range(max(1, count // 2)):
        task = asyncio.create_task(leaked_task(i))
        _leaked_tasks.append(task)
        leaked_tasks.append(f"leaked_task_{i}")
    
    # Create leaked timers
    loop = asyncio.get_event_loop()
    for i in range(max(1, count // 2)):
        timer = loop.call_later(300, leaked_timer_callback, i)  # 5 minute timer
        _leaked_timers.append(timer)
        leaked_timers.append(f"leaked_timer_{i}")
    
    # GROUND-TRUTH TRACKING: Record injected asyncio leaks
    total_asyncio_leaks = len(leaked_tasks) + len(leaked_timers)
    _ground_truth_leaks['asyncio_tasks_injected'] += len(leaked_tasks)
    _ground_truth_leaks['asyncio_timers_injected'] += len(leaked_timers)
    _ground_truth_leaks['total_injected'] += total_asyncio_leaks
    _ground_truth_leaks['injection_log'].append({
        'timestamp': time.time(),
        'type': 'asyncio_leaks',
        'tasks_count': len(leaked_tasks),
        'timers_count': len(leaked_timers),
        'total_count': total_asyncio_leaks
    })
    
    return {
        "leaked_tasks": leaked_tasks,
        "leaked_timers": leaked_timers,
        "count": len(leaked_tasks) + len(leaked_timers),
        "total_leaked_tasks": len(_leaked_tasks),
        "total_leaked_timers": len(_leaked_timers),
        "ground_truth_injected_tasks": _ground_truth_leaks['asyncio_tasks_injected'],
        "ground_truth_injected_timers": _ground_truth_leaks['asyncio_timers_injected']
    }

@app.get("/test/leak-status")
async def leak_status():
    """Get current status of all generated leaks with ground-truth tracking"""
    
    # Try to get MemGuard detection stats for comparison
    memguard_stats = {}
    try:
        import memguard
        if memguard.is_protecting():
            report = memguard.get_report()
            # Count findings by pattern for detection rate calculation
            pattern_counts = {}
            for finding in report.findings:
                pattern = finding.pattern
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            memguard_stats = {
                'total_findings': len(report.findings),
                'findings_by_pattern': pattern_counts
            }
    except:
        pass
    
    return {
        # Current leak storage counts
        "file_leaks": len(_leaked_files),
        "socket_leaks": len(_leaked_sockets), 
        "cache_keys": len(_leaked_caches),
        "cache_total_mb": sum(cache["size_mb"] for cache in _leaked_caches.values()),
        "asyncio_tasks": len(_leaked_tasks),
        "asyncio_timers": len(_leaked_timers),
        
        # GROUND-TRUTH METRICS for detection rate calculation
        "ground_truth": _ground_truth_leaks.copy(),
        
        # MemGuard detection stats for comparison
        "memguard_detections": memguard_stats,
        
        # Summary
        "summary": f"Injected: {_ground_truth_leaks['total_injected']} leaks | Active: {len(_leaked_files) + len(_leaked_sockets) + len(_leaked_caches) + len(_leaked_tasks) + len(_leaked_timers)} resources"
    }

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    print("🛑 TaskFlow API shutting down...")
    
    # Clean up database connections
    for conn_info in db_connections:
        try:
            conn_info['conn'].close()
        except:
            pass
    
    # Stop MemGuard
    try:
        import memguard
        if memguard.is_protecting():
            memguard.stop()
            print("🛡️  MemGuard monitoring stopped")
    except:
        pass
    
    print("✅ Shutdown complete")

# ======================================
# MAIN
# ======================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 TaskFlow API - Real-World MemGuard Integration Demo")
    print("=" * 60)
    print()
    print("This is a realistic Python web API that demonstrates:")
    print("✅ Real memory usage patterns developers create")
    print("✅ MemGuard Pro monitoring with auto-cleanup")
    print("✅ Production-ready monitoring integration")
    print()
    print("📍 API will run at: http://localhost:8000")
    print("📖 Documentation: http://localhost:8000/docs")
    print("🛡️  MemGuard Status: http://localhost:8000/memguard/status")
    print("📊 MemGuard Report: http://localhost:8000/memguard/report")
    print()
    print("🔥 Try the stress test: POST http://localhost:8000/stress-test/")
    print("   This will trigger memory usage and show MemGuard in action!")
    print()
    
    uvicorn.run(
        "taskflow_api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,  # Disabled for stable extended testing
        log_level="info",
        # Extended testing stability improvements
        timeout_keep_alive=30,  # Keep connections alive longer
        timeout_graceful_shutdown=30,  # More time for graceful shutdown
        limit_concurrency=1000,  # Handle more concurrent requests
        limit_max_requests=100000  # Allow more total requests before restart
    )