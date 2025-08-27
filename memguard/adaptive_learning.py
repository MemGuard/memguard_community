#!/usr/bin/env python3
"""
Adaptive Learning System for MemGuard
Learns from file usage patterns to improve protection decisions for unknown file types.
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict, deque
import logging

_logger = logging.getLogger(__name__)


class AdaptiveLearningEngine:
    """
    Learns from file behavior patterns to make better protection decisions.
    Adapts to unknown file types and user-specific usage patterns.
    """
    
    def __init__(self, learning_data_path: Optional[Path] = None):
        self.learning_data_path = learning_data_path or Path.home() / '.memguard' / 'adaptive_learning.json'
        self.lock = threading.RLock()
        
        # Learning data structures with PRODUCTION-OPTIMIZED defaults
        self.extension_behaviors: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_files': 0,
            'closed_by_cleanup': 0,
            'remained_protected': 0,
            'average_lifetime': 0.0,
            'access_patterns': [],
            'size_patterns': [],
            'protection_effectiveness': 0.3  # Start more aggressive (lower protection = higher cleanup)
        })
        
        # Initialize with smart defaults for common file types
        self._initialize_smart_defaults()
        
        self.process_patterns: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'file_extensions_used': set(),
            'typical_file_lifetime': 0.0,
            'cleanup_tolerance': 0.5
        })
        
        self.user_feedback: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'false_positives': 0,  # Files incorrectly cleaned
            'false_negatives': 0,  # Files that should have been cleaned
            'satisfaction_score': 0.5
        })
        
        # Recent activity tracking
        self.recent_activity = deque(maxlen=1000)
        
        # Load existing learning data
        self._load_learning_data()
    
    def _initialize_smart_defaults(self):
        """Initialize with smart defaults for common file types."""
        # Temporary files - very low protection (high cleanup likelihood)
        temp_extensions = ['.tmp', '.temp', '.cache', '.bak', '.old', '.swap', '~', '.swp']
        for ext in temp_extensions:
            self.extension_behaviors[ext]['protection_effectiveness'] = 0.1  # Very cleanable
            self.extension_behaviors[ext]['average_lifetime'] = 300.0  # 5 minutes typical
        
        # Log files - moderate protection
        log_extensions = ['.log', '.out', '.err']
        for ext in log_extensions:
            self.extension_behaviors[ext]['protection_effectiveness'] = 0.25  # Moderately cleanable
            self.extension_behaviors[ext]['average_lifetime'] = 1800.0  # 30 minutes typical
        
        # System files - high protection
        system_extensions = ['.exe', '.dll', '.sys', '.bin', '.so', '.dylib']
        for ext in system_extensions:
            self.extension_behaviors[ext]['protection_effectiveness'] = 0.9  # Highly protected
            self.extension_behaviors[ext]['average_lifetime'] = 86400.0  # 24 hours typical
        
        # Source files - high protection
        source_extensions = ['.py', '.js', '.html', '.css', '.json', '.xml', '.c', '.cpp', '.h']
        for ext in source_extensions:
            self.extension_behaviors[ext]['protection_effectiveness'] = 0.8  # Highly protected
            self.extension_behaviors[ext]['average_lifetime'] = 3600.0  # 1 hour typical
        
        # Data/config files - moderate protection
        data_extensions = ['.txt', '.dat', '.cfg', '.conf', '.ini', '.yaml', '.yml']
        for ext in data_extensions:
            self.extension_behaviors[ext]['protection_effectiveness'] = 0.4  # Moderate protection
            self.extension_behaviors[ext]['average_lifetime'] = 1800.0  # 30 minutes typical
    
    def learn_from_file_behavior(self, extension: str, file_size: int, 
                                lifetime_seconds: float, was_cleaned: bool,
                                access_count: int = 0, process_name: str = None):
        """Learn from observed file behavior patterns."""
        with self.lock:
            ext_data = self.extension_behaviors[extension.lower()]
            
            # Update statistics
            ext_data['total_files'] += 1
            if was_cleaned:
                ext_data['closed_by_cleanup'] += 1
            else:
                ext_data['remained_protected'] += 1
            
            # Update average lifetime (exponential moving average)
            alpha = 0.1  # Learning rate
            ext_data['average_lifetime'] = (
                alpha * lifetime_seconds + 
                (1 - alpha) * ext_data['average_lifetime']
            )
            
            # Track access patterns
            ext_data['access_patterns'].append({
                'access_count': access_count,
                'lifetime': lifetime_seconds,
                'timestamp': time.time()
            })
            
            # Keep only recent patterns (last 100)
            if len(ext_data['access_patterns']) > 100:
                ext_data['access_patterns'] = ext_data['access_patterns'][-100:]
            
            # Track size patterns
            ext_data['size_patterns'].append(file_size)
            if len(ext_data['size_patterns']) > 100:
                ext_data['size_patterns'] = ext_data['size_patterns'][-100:]
            
            # Update protection effectiveness
            if ext_data['total_files'] >= 5:  # Need minimum data for learning
                cleanup_rate = ext_data['closed_by_cleanup'] / ext_data['total_files']
                
                # Extensions with high cleanup rates are probably safe to clean
                # Extensions with low cleanup rates are probably important
                if cleanup_rate > 0.8:
                    ext_data['protection_effectiveness'] = min(0.9, 0.3)  # Low protection needed
                elif cleanup_rate < 0.2:
                    ext_data['protection_effectiveness'] = max(0.1, 0.8)  # High protection needed
                else:
                    ext_data['protection_effectiveness'] = 0.5  # Neutral
            
            # Learn process patterns
            if process_name:
                proc_data = self.process_patterns[process_name]
                proc_data['file_extensions_used'].add(extension.lower())
                proc_data['typical_file_lifetime'] = (
                    alpha * lifetime_seconds + 
                    (1 - alpha) * proc_data['typical_file_lifetime']
                )
            
            # Record recent activity
            self.recent_activity.append({
                'extension': extension,
                'size': file_size,
                'lifetime': lifetime_seconds,
                'cleaned': was_cleaned,
                'process': process_name,
                'timestamp': time.time()
            })
    
    def get_protection_recommendation(self, extension: str, file_size: int,
                                   age_seconds: float, process_name: str = None) -> float:
        """
        Get adaptive learning protection recommendation for unknown file types.
        Returns 0.0 (definitely clean) to 1.0 (definitely protect).
        """
        with self.lock:
            ext = extension.lower()
            
            # Check if we have learned data for this extension
            if ext in self.extension_behaviors:
                ext_data = self.extension_behaviors[ext]
                
                # Use learned protection effectiveness as base
                base_protection = ext_data['protection_effectiveness']
                
                # Adjust based on file characteristics vs learned patterns
                size_adjustment = self._analyze_size_pattern(file_size, ext_data['size_patterns'])
                age_adjustment = self._analyze_age_pattern(age_seconds, ext_data['average_lifetime'])
                
                # Combine factors
                protection_score = (base_protection + size_adjustment + age_adjustment) / 3.0
                
                return max(0.0, min(1.0, protection_score))
            
            # No specific data for this extension - use similarity analysis
            similar_extensions = self._find_similar_extensions(ext)
            if similar_extensions:
                # Average protection scores from similar extensions
                similar_scores = [
                    self.extension_behaviors[sim_ext]['protection_effectiveness']
                    for sim_ext in similar_extensions
                ]
                return sum(similar_scores) / len(similar_scores)
            
            # Unknown extension with no similar patterns - use conservative heuristics
            return self._conservative_unknown_file_analysis(extension, file_size, age_seconds)
    
    def _analyze_size_pattern(self, current_size: int, historical_sizes: List[int]) -> float:
        """Analyze if file size matches learned patterns."""
        if not historical_sizes:
            return 0.5
        
        # Calculate size percentiles
        sorted_sizes = sorted(historical_sizes)
        median_size = sorted_sizes[len(sorted_sizes) // 2]
        
        # Files significantly larger than typical might be more important
        if current_size > median_size * 10:
            return 0.8  # Strongly protect
        elif current_size > median_size * 2:
            return 0.6  # Moderately protect
        elif current_size < median_size * 0.1:
            return 0.3  # Less protection (might be temp/empty)
        else:
            return 0.5  # Neutral
    
    def _analyze_age_pattern(self, current_age: float, typical_lifetime: float) -> float:
        """Analyze if file age suggests abandonment vs normal usage."""
        if typical_lifetime <= 0:
            return 0.5
        
        age_ratio = current_age / typical_lifetime
        
        # Files much older than typical lifetime are likely abandoned
        if age_ratio > 5.0:
            return 0.2  # Likely abandoned, less protection needed
        elif age_ratio > 2.0:
            return 0.3  # Possibly abandoned
        elif age_ratio < 0.5:
            return 0.7  # Still within normal lifetime, protect
        else:
            return 0.5  # Neutral
    
    def _find_similar_extensions(self, target_ext: str) -> List[str]:
        """Find extensions with similar behavior patterns."""
        target_ext = target_ext.lower()
        similar = []
        
        # Simple similarity heuristics
        for ext in self.extension_behaviors:
            if len(ext) == len(target_ext):  # Same length
                differences = sum(1 for a, b in zip(ext, target_ext) if a != b)
                if differences <= 1:  # At most 1 character different
                    similar.append(ext)
            elif abs(len(ext) - len(target_ext)) == 1:  # Length differs by 1
                # Check for simple additions/deletions
                if target_ext in ext or ext in target_ext:
                    similar.append(ext)
        
        return similar[:5]  # Limit to top 5 similar extensions
    
    def _conservative_unknown_file_analysis(self, extension: str, file_size: int, age_seconds: float) -> float:
        """PRODUCTION-OPTIMIZED analysis for completely unknown file types."""
        # START MORE AGGRESSIVE: 0.3 instead of 0.5 for better cleanup rates
        protection_score = 0.3  # Start with lower protection (higher cleanup likelihood)
        
        # File size heuristics - enhanced for production
        if file_size > 500 * 1024 * 1024:  # 500MB+ - definitely important
            protection_score += 0.4
        elif file_size > 100 * 1024 * 1024:  # 100MB+ - probably important  
            protection_score += 0.2
        elif file_size > 10 * 1024 * 1024:   # 10MB+ - might be important
            protection_score += 0.1
        elif file_size == 0:
            protection_score -= 0.4  # Empty files - very likely temp/abandoned
        elif file_size < 1024:  # Very small files
            protection_score -= 0.1  # Often config/temp files
        
        # IMPROVED Age heuristics - more realistic for production
        if age_seconds < 10.0:
            protection_score += 0.3  # Very new files, be very careful
        elif age_seconds < 60.0:  # 1 minute
            protection_score += 0.1  # New files, be careful
        elif age_seconds > 1800.0:  # 30 minutes - very likely abandoned
            protection_score -= 0.4
        elif age_seconds > 900.0:   # 15 minutes - likely abandoned
            protection_score -= 0.3
        elif age_seconds > 300.0:   # 5 minutes - possibly abandoned
            protection_score -= 0.2
        
        # ENHANCED extension pattern heuristics
        ext_lower = extension.lower()
        
        # Strong temporary indicators - aggressive cleanup
        if any(temp_indicator in ext_lower for temp_indicator in ['.tmp', '.temp', '.cache', '.bak', '.old', '.swap', '~']):
            protection_score -= 0.5  # Very likely safe to clean
        
        # Log files - context dependent
        elif '.log' in ext_lower:
            if age_seconds > 300.0:  # Old log files
                protection_score -= 0.3
            else:
                protection_score += 0.1  # Active log files
        
        # System/important extensions
        elif ext_lower in ['.exe', '.dll', '.sys', '.bin', '.so', '.dylib']:
            protection_score += 0.4  # System files - protect
        elif ext_lower in ['.py', '.js', '.html', '.css', '.json', '.xml']:
            protection_score += 0.2  # Source files - protect
        elif ext_lower in ['.txt', '.dat', '.cfg', '.conf', '.ini']:
            protection_score += 0.1  # Config/data files - mild protection
        
        # Extension length heuristics
        elif len(extension) <= 2:  # Very short extensions (.c, .h, etc)
            protection_score += 0.1  # Often important
        elif len(extension) > 6:  # Very long extensions
            protection_score += 0.2  # Often proprietary/specialized formats
        
        return max(0.05, min(0.95, protection_score))
    
    def record_user_feedback(self, extension: str, feedback_type: str, description: str = None):
        """Record user feedback about protection decisions."""
        with self.lock:
            ext = extension.lower()
            feedback_data = self.user_feedback[ext]
            
            if feedback_type == 'false_positive':  # Incorrectly cleaned important file
                feedback_data['false_positives'] += 1
                feedback_data['satisfaction_score'] = max(0.0, feedback_data['satisfaction_score'] - 0.1)
                
                # Immediately adjust protection for this extension
                if ext in self.extension_behaviors:
                    self.extension_behaviors[ext]['protection_effectiveness'] = min(1.0, 
                        self.extension_behaviors[ext]['protection_effectiveness'] + 0.2)
                        
            elif feedback_type == 'false_negative':  # Should have cleaned but didn't
                feedback_data['false_negatives'] += 1
                feedback_data['satisfaction_score'] = max(0.0, feedback_data['satisfaction_score'] - 0.05)
                
                # Adjust protection downward
                if ext in self.extension_behaviors:
                    self.extension_behaviors[ext]['protection_effectiveness'] = max(0.0,
                        self.extension_behaviors[ext]['protection_effectiveness'] - 0.1)
            
            elif feedback_type == 'positive':  # Correct decision
                feedback_data['satisfaction_score'] = min(1.0, feedback_data['satisfaction_score'] + 0.05)
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get current learning statistics."""
        with self.lock:
            return {
                'extensions_learned': len(self.extension_behaviors),
                'total_observations': sum(data['total_files'] for data in self.extension_behaviors.values()),
                'processes_observed': len(self.process_patterns),
                'recent_activity_count': len(self.recent_activity),
                'top_protected_extensions': sorted(
                    [(ext, data['protection_effectiveness']) for ext, data in self.extension_behaviors.items()],
                    key=lambda x: x[1], reverse=True
                )[:10],
                'learning_data_path': str(self.learning_data_path)
            }
    
    def _load_learning_data(self):
        """Load previously saved learning data."""
        try:
            if self.learning_data_path.exists():
                with open(self.learning_data_path, 'r') as f:
                    data = json.load(f)
                
                # Convert loaded data back to defaultdicts
                for ext, ext_data in data.get('extension_behaviors', {}).items():
                    self.extension_behaviors[ext] = ext_data
                    # Convert sets back from lists
                    if 'file_extensions_used' in ext_data:
                        ext_data['file_extensions_used'] = set(ext_data['file_extensions_used'])
                
                for proc, proc_data in data.get('process_patterns', {}).items():
                    self.process_patterns[proc] = proc_data
                    if 'file_extensions_used' in proc_data:
                        proc_data['file_extensions_used'] = set(proc_data['file_extensions_used'])
                
                for ext, feedback_data in data.get('user_feedback', {}).items():
                    self.user_feedback[ext] = feedback_data
                
                _logger.info(f"Loaded adaptive learning data: {len(self.extension_behaviors)} extensions")
                
        except Exception as e:
            _logger.warning(f"Could not load adaptive learning data: {e}")
    
    def save_learning_data(self):
        """Save current learning data to disk."""
        try:
            # Ensure directory exists
            self.learning_data_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert sets to lists for JSON serialization
            data_to_save = {}
            
            with self.lock:
                # Extension behaviors
                ext_behaviors = {}
                for ext, ext_data in self.extension_behaviors.items():
                    ext_data_copy = ext_data.copy()
                    if 'file_extensions_used' in ext_data_copy:
                        ext_data_copy['file_extensions_used'] = list(ext_data_copy['file_extensions_used'])
                    ext_behaviors[ext] = ext_data_copy
                data_to_save['extension_behaviors'] = ext_behaviors
                
                # Process patterns
                proc_patterns = {}
                for proc, proc_data in self.process_patterns.items():
                    proc_data_copy = proc_data.copy()
                    if 'file_extensions_used' in proc_data_copy:
                        proc_data_copy['file_extensions_used'] = list(proc_data_copy['file_extensions_used'])
                    proc_patterns[proc] = proc_data_copy
                data_to_save['process_patterns'] = proc_patterns
                
                # User feedback
                data_to_save['user_feedback'] = dict(self.user_feedback)
                
                # Metadata
                data_to_save['metadata'] = {
                    'last_saved': time.time(),
                    'version': '1.0',
                    'total_observations': sum(data['total_files'] for data in self.extension_behaviors.values())
                }
            
            with open(self.learning_data_path, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            _logger.info(f"Saved adaptive learning data to {self.learning_data_path}")
            
        except Exception as e:
            _logger.error(f"Could not save adaptive learning data: {e}")


# Global learning engine instance
_learning_engine: Optional[AdaptiveLearningEngine] = None
_learning_lock = threading.Lock()


def get_learning_engine() -> AdaptiveLearningEngine:
    """Get the global adaptive learning engine instance."""
    global _learning_engine
    
    if _learning_engine is None:
        with _learning_lock:
            if _learning_engine is None:
                _learning_engine = AdaptiveLearningEngine()
    
    return _learning_engine


def learn_from_cleanup_decision(extension: str, file_size: int, lifetime_seconds: float,
                              was_cleaned: bool, process_name: str = None):
    """Record a cleanup decision for learning."""
    engine = get_learning_engine()
    engine.learn_from_file_behavior(extension, file_size, lifetime_seconds, 
                                   was_cleaned, process_name=process_name)


def get_adaptive_protection_score(extension: str, file_size: int, age_seconds: float,
                                process_name: str = None) -> float:
    """Get adaptive learning protection recommendation."""
    engine = get_learning_engine()
    return engine.get_protection_recommendation(extension, file_size, age_seconds, process_name)


def save_learning_state():
    """Save current learning state to disk."""
    if _learning_engine:
        _learning_engine.save_learning_data()