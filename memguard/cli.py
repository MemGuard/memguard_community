#!/usr/bin/env python3
"""
MemGuard CLI Interface

Command-line interface for MemGuard Pro memory leak detection.
"""

import argparse
import sys
import json
import time
import subprocess
from pathlib import Path

from . import protect, stop, analyze, get_status, __version__


def create_parser():
    """Create the argument parser for MemGuard CLI."""
    parser = argparse.ArgumentParser(
        prog='memguard',
        description='MemGuard Pro - Advanced Memory Leak Detection and Auto-Cleanup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  memguard start --threshold 50 --sample-rate 0.1
  memguard analyze --output report.json
  memguard stop
  memguard status
  
Upgrade to Pro:
  memguard upgrade --pro
  memguard start --license YOUR-LICENSE-KEY --auto-cleanup
  memguard analyze --cost-analysis
        """
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'%(prog)s {__version__}'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start memory leak monitoring')
    start_parser.add_argument('--threshold', '-t', type=float, default=100.0,
                             help='Memory threshold in MB (default: 100)')
    start_parser.add_argument('--sample-rate', '-s', type=float, default=0.1,
                             help='Sampling rate 0.0-1.0 (default: 0.1)')
    start_parser.add_argument('--patterns', '-p', nargs='+', 
                             default=['handles', 'caches'],
                             choices=['handles', 'caches', 'cycles', 'timers', 'listeners'],
                             help='Leak patterns to detect')
    start_parser.add_argument('--poll-interval', type=float, default=30.0,
                             help='Polling interval in seconds (default: 30)')
    start_parser.add_argument('--license', '-l', type=str,
                             help='Pro license key for advanced features')
    start_parser.add_argument('--auto-cleanup', action='store_true',
                             help='Enable auto-cleanup (requires Pro license)')
    start_parser.add_argument('--aggressive', action='store_true',
                             help='Enable aggressive detection mode')
    start_parser.add_argument('--background', action='store_true', default=True,
                             help='Run in background mode (default: True)')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop memory leak monitoring')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze current memory state')
    analyze_parser.add_argument('--output', '-o', type=str,
                               help='Output file for report (JSON format)')
    analyze_parser.add_argument('--format', choices=['json', 'text'], default='text',
                               help='Output format (default: text)')
    analyze_parser.add_argument('--cost-analysis', action='store_true',
                               help='Include detailed cost analysis')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show monitoring status')
    status_parser.add_argument('--json', action='store_true',
                              help='Output in JSON format')
    
    # Upgrade command
    upgrade_parser = subparsers.add_parser('upgrade', help='Upgrade MemGuard')
    upgrade_parser.add_argument('--pro', action='store_true',
                               help='Upgrade to Pro version with Firebase licensing')
    upgrade_parser.add_argument('--check', action='store_true',
                               help='Check for available upgrades')
    
    return parser


def format_report_text(report):
    """Format report for text output."""
    lines = []
    lines.append("=" * 60)
    lines.append("MemGuard Analysis Report")
    lines.append("=" * 60)
    
    # Health Score (NEW - always shown)
    health_score = getattr(report, 'health_score', 100)
    health_status = getattr(report, 'health_status', 'Unknown')
    health_grade = getattr(report, 'health_grade', 'A')
    
    lines.append(f"🏥 Memory Health Score: {health_score}/100 ({health_grade}) - {health_status}")
    
    # Cost analysis (always available now)
    if hasattr(report, 'estimated_monthly_cost_usd') and report.estimated_monthly_cost_usd > 0:
        lines.append(f"💰 Monthly Waste: ${report.estimated_monthly_cost_usd:.2f}")
    
    # Basic info
    if hasattr(report, 'scan_duration_ms'):
        lines.append(f"⚡ Scan Duration: {report.scan_duration_ms:.1f} ms")
    if hasattr(report, 'memory_current_mb'):
        lines.append(f"💾 Current Memory: {report.memory_current_mb:.1f} MB")
    
    lines.append("")
    
    # Findings with prevention tips
    lines.append(f"🔍 Findings: {len(report.findings)}")
    lines.append("-" * 20)
    
    if not report.findings:
        lines.append("✅ No memory leaks detected - Great job!")
    else:
        for i, finding in enumerate(report.findings, 1):
            # Severity emoji
            severity_emoji = {
                'low': '🟡',
                'medium': '🟠', 
                'high': '🔴',
                'critical': '🚨'
            }.get(finding.severity.value, '⚪')
            
            lines.append(f"{i}. {severity_emoji} {finding.pattern.upper()}: {finding.detail}")
            lines.append(f"   📍 Location: {finding.location}")
            lines.append(f"   📊 Severity: {finding.severity.value.title()}")
            lines.append(f"   🎯 Confidence: {finding.confidence:.0%}")
            
            if hasattr(finding, 'size_mb') and finding.size_mb:
                lines.append(f"   📏 Size: {finding.size_mb:.2f} MB")
            
            # Prevention tip
            if hasattr(report, 'get_prevention_tip'):
                tip = report.get_prevention_tip(finding)
                lines.append(f"   {tip}")
            
            lines.append("")
    
    # Pro upgrade prompt (contextual)
    if hasattr(report, 'should_suggest_pro_upgrade') and report.should_suggest_pro_upgrade():
        lines.append("🆙 " + "=" * 58)
        upgrade_msg = report.get_pro_upgrade_message()
        lines.append(upgrade_msg)
        lines.append("=" * 60)
        lines.append("")
    
    # License info
    if hasattr(report, 'license_type'):
        license_emoji = "✨" if report.license_type == 'PRO' else "🆓"
        lines.append(f"{license_emoji} License: {report.license_type}")
        
    if hasattr(report, 'pro_features_enabled') and report.pro_features_enabled:
        lines.append(f"🚀 Pro Features: {', '.join(report.pro_features_enabled)}")
    
    return "\n".join(lines)


def format_status_text(status):
    """Format status for text output with production-grade overhead visibility."""
    lines = []
    lines.append("MemGuard Production Status")
    lines.append("=" * 26)
    
    # Core protection status
    protection_emoji = "🛡️" if status.get('is_protecting') else "⏹️"
    lines.append(f"{protection_emoji} Protection: {'ACTIVE' if status.get('is_protecting') else 'STOPPED'}")
    
    # CRITICAL: Overhead visibility for customer self-verification
    if 'performance_stats' in status:
        perf = status['performance_stats']
        overhead_pct = perf.get('overhead_percentage', 0.0)
        
        # Color coding for overhead (production thresholds)
        if overhead_pct < 3.0:
            overhead_emoji = "✅"  # Excellent
        elif overhead_pct < 5.0:
            overhead_emoji = "🟡"  # Acceptable
        elif overhead_pct < 10.0:
            overhead_emoji = "🟠"  # Warning
        else:
            overhead_emoji = "🔴"  # Critical
            
        lines.append(f"{overhead_emoji} Overhead: {overhead_pct:.2f}% (scan: {perf.get('avg_scan_duration_ms', 0):.1f}ms)")
        
        # Memory tracking
        memory_current = perf.get('memory_current_mb', 0.0)
        memory_baseline = perf.get('memory_baseline_mb', 0.0)
        memory_growth = perf.get('memory_growth_mb', 0.0)
        
        if memory_growth > 0:
            lines.append(f"📈 Memory: {memory_current:.1f}MB (+{memory_growth:.1f}MB from baseline)")
        else:
            lines.append(f"💾 Memory: {memory_current:.1f}MB (stable)")
            
        # Scan performance
        total_scans = perf.get('total_scans', 0)
        scan_freq = perf.get('scan_frequency_hz', 0.0)
        lines.append(f"🔍 Scans: {total_scans} total @ {scan_freq:.1f}Hz")
        
        # Findings summary
        total_findings = perf.get('total_findings', 0)
        if total_findings > 0:
            lines.append(f"⚠️  Findings: {total_findings} issues detected")
        else:
            lines.append("✅ Findings: No issues detected")
    
    lines.append("")
    
    # Uptime and health
    if 'uptime_seconds' in status:
        uptime_hours = status['uptime_seconds'] / 3600
        if uptime_hours < 1:
            uptime_str = f"{status['uptime_seconds']:.0f}s"
        elif uptime_hours < 24:
            uptime_str = f"{uptime_hours:.1f}h"
        else:
            uptime_days = uptime_hours / 24
            uptime_str = f"{uptime_days:.1f}d"
        lines.append(f"⏰ Uptime: {uptime_str}")
    
    # Component status with versions (observability)
    if 'guards' in status and status['guards']:
        lines.append("🔧 Guards:")
        for pattern_name, guard_info in status['guards'].items():
            status_emoji = "✅" if guard_info.get('active') else "⏹️"
            version = guard_info.get('version', '1.0.0')
            guard_status = guard_info.get('status', 'unknown')
            lines.append(f"   {status_emoji} {pattern_name} v{version} ({guard_status})")
    
    if 'detectors' in status and status['detectors']:
        lines.append("🕵️ Detectors:")
        for pattern_name, detector_info in status['detectors'].items():
            status_emoji = "✅" if detector_info.get('active') else "⏹️"
            version = detector_info.get('version', '1.0.0')
            detector_status = detector_info.get('status', 'unknown')
            lines.append(f"   {status_emoji} {pattern_name} v{version} ({detector_status})")
    
    # Environment info for troubleshooting
    if 'environment' in status:
        env = status['environment']
        lines.append("")
        lines.append(f"🖥️  Environment: {env.get('platform', 'unknown')}")
        lines.append(f"🐍 Python: {env.get('python_version', 'unknown')}")
        lines.append(f"📍 Process: {env.get('process_name', 'unknown')}")
    
    # Schema version for compatibility
    if 'schema_version' in status:
        lines.append(f"📋 Status API: v{status['schema_version']}")
    
    return "\n".join(lines)


def cmd_start(args):
    """Handle start command."""
    try:
        kwargs = {
            'threshold_mb': args.threshold,
            'sample_rate': args.sample_rate,
            'patterns': tuple(args.patterns),
            'poll_interval_s': args.poll_interval,
            'background': args.background,
        }
        
        if args.license:
            if not check_pro_features_available():
                print("⚠️  License key provided but Pro features not installed")
                suggest_pro_upgrade()
                return 1
            kwargs['license_key'] = args.license
        
        if args.auto_cleanup:
            if not check_pro_features_available():
                print("⚠️  Auto-cleanup requires Pro features")
                suggest_pro_upgrade()
                return 1
            kwargs['auto_cleanup'] = {pattern: True for pattern in args.patterns}
        
        if args.aggressive:
            if not check_pro_features_available():
                print("⚠️  Aggressive mode requires Pro features")
                suggest_pro_upgrade()
                return 1
            kwargs['aggressive_mode'] = True
        
        protect(**kwargs)
        
        print(f"✅ MemGuard started successfully")
        print(f"   Threshold: {args.threshold} MB")
        print(f"   Sample Rate: {args.sample_rate}")
        print(f"   Patterns: {', '.join(args.patterns)}")
        
        if args.license:
            print(f"   License: Pro")
        if args.auto_cleanup:
            print(f"   Auto-cleanup: Enabled")
        
    except Exception as e:
        print(f"❌ Failed to start MemGuard: {e}")
        return 1
    
    return 0


def cmd_stop(args):
    """Handle stop command."""
    try:
        stop()
        print("✅ MemGuard stopped successfully")
    except Exception as e:
        print(f"❌ Failed to stop MemGuard: {e}")
        return 1
    
    return 0


def cmd_analyze(args):
    """Handle analyze command."""
    try:
        report = analyze()
        
        if args.format == 'json':
            # Convert report to JSON-serializable format
            report_dict = {
                'created_at': getattr(report, 'created_at', time.time()),
                'findings': [
                    {
                        'pattern': f.pattern,
                        'location': f.location,
                        'detail': f.detail,
                        'severity': f.severity,
                        'confidence': f.confidence,
                        'size_mb': getattr(f, 'size_mb', 0),
                    }
                    for f in report.findings
                ],
                'scan_duration_ms': getattr(report, 'scan_duration_ms', 0),
                'memory_current_mb': getattr(report, 'memory_current_mb', 0),
            }
            
            # Add Pro features if available
            if hasattr(report, 'license_type'):
                report_dict['license_type'] = report.license_type
            if hasattr(report, 'estimated_monthly_cost_usd'):
                report_dict['estimated_monthly_cost_usd'] = report.estimated_monthly_cost_usd
            if hasattr(report, 'pro_features_enabled'):
                report_dict['pro_features_enabled'] = report.pro_features_enabled
            
            output = json.dumps(report_dict, indent=2)
        else:
            output = format_report_text(report)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"✅ Report saved to {args.output}")
        else:
            print(output)
        
    except Exception as e:
        print(f"❌ Failed to analyze: {e}")
        return 1
    
    return 0


def cmd_status(args):
    """Handle status command."""
    try:
        status = get_status()
        
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(format_status_text(status))
        
    except Exception as e:
        print(f"❌ Failed to get status: {e}")
        return 1
    
    return 0


def cmd_upgrade(args):
    """Handle upgrade command."""
    if args.check:
        print("🔍 Checking for available upgrades...")
        
        # Check if Pro dependencies are available
        try:
            import firebase_admin
            print("✅ Pro features already available")
            return 0
        except ImportError:
            print("📦 Pro upgrade available:")
            print("   - Advanced detection algorithms")
            print("   - Auto-cleanup functionality (saves $$$ automatically)")
            print("   - Custom patterns & detection rules")
            print("   - Enterprise reporting")
            print("   - Strategic insights & architectural analysis")
            print("   - Priority support")
            print("\n💡 Run 'memguard upgrade --pro' to upgrade")
            return 0
    
    if args.pro:
        print("🚀 Upgrading to MemGuard Pro...")
        
        try:
            # Install Pro dependencies
            print("📦 Installing Pro dependencies...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 'memguard[pro]'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Pro upgrade completed successfully!")
                print("\n🎉 Pro features now available:")
                print("   ✅ Advanced detection algorithms")
                print("   ✅ Auto-cleanup functionality (saves $$$ automatically)")
                print("   ✅ Custom patterns & detection rules")
                print("   ✅ Enterprise reporting")
                print("   ✅ Strategic insights & architectural analysis")
                print("   ✅ Priority support")
                
                print("\n💡 Next steps:")
                print("   1. Get your Pro subscription: https://memguard.net/")
                print("   2. Start with Pro features:")
                print("      memguard start --license YOUR-LICENSE-KEY --auto-cleanup")
                
                # Test Pro import
                try:
                    import firebase_admin
                    print("   ✅ Pro dependencies verified")
                except ImportError:
                    print("   ⚠️  Pro dependencies may need manual installation")
                
            else:
                print(f"❌ Upgrade failed: {result.stderr}")
                print("\n💡 Manual upgrade:")
                print("   pip install memguard[pro]")
                return 1
                
        except Exception as e:
            print(f"❌ Upgrade failed: {e}")
            print("\n💡 Manual upgrade:")
            print("   pip install memguard[pro]")
            return 1
    
    else:
        print("💡 Available upgrade options:")
        print("   --pro     Upgrade to Pro version")
        print("   --check   Check for available upgrades")
        print("\nExample: memguard upgrade --pro")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Dispatch to command handlers
    handlers = {
        'start': cmd_start,
        'stop': cmd_stop,
        'analyze': cmd_analyze,
        'status': cmd_status,
        'upgrade': cmd_upgrade,
    }
    
    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1


def check_pro_features_available():
    """Check if Pro features are available."""
    try:
        import firebase_admin
        return True
    except ImportError:
        return False


def suggest_pro_upgrade():
    """Suggest Pro upgrade if features are used without Pro license."""
    print("\n💡 This feature requires MemGuard Pro")
    print("   Upgrade with: memguard upgrade --pro")
    print("   Or install manually: pip install memguard[pro]")
    print("   Get your subscription: https://memguard.net/")


if __name__ == '__main__':
    sys.exit(main())