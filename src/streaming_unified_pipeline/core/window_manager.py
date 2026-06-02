#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Window Manager for Unified Streaming Evaluation Pipeline

This module provides unified windowing functionality for all tasks:
- Date-based sliding windows (Task-1, Task-3)
- Count-based windows (Task-2)
- Support for incremental processing (Task-1)
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Iterator, Optional, Union
from collections import defaultdict

class WindowManager:
    """
    Unified window management for streaming evaluation tasks.
    """
    
    def __init__(self, 
                 window_type: str = "date",  # "date", "count", "incremental"
                 window_size: Union[int, float] = 5,  # days for date, count for count
                 stride: Optional[Union[int, float]] = None,  # stride for sliding windows (auto if None)
                 overlap: bool = True):  # whether windows can overlap
        """
        Initialize WindowManager.
        
        Args:
            window_type: Type of windowing ("date", "count", "incremental")
            window_size: Size of each window (days for date-based, count for count-based)
            stride: Stride between windows (for sliding windows)
            overlap: Whether windows can overlap
        """
        self.window_type = window_type
        self.window_size = window_size
        
        # Auto-set stride based on window type if not specified
        if stride is None:
            if window_type == "date":
                self.stride = 1  # 1 day stride for date windows
            elif window_type == "count":
                self.stride = 1  # 1 event stride for count windows (sliding)
            elif window_type == "incremental":
                self.stride = 1  # Not used, but set for consistency
        else:
            self.stride = stride
            
        self.overlap = overlap
        
        logging.info(f"WindowManager initialized: type={window_type}, size={window_size}, stride={stride}")
    
    def create_windows(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create windows from events based on the configured window type.
        
        Args:
            events: List of event dictionaries with 'date' and other fields
            
        Returns:
            List of window dictionaries
        """
        if not events:
            return []
        
        if self.window_type == "date" or self.window_type == "sliding":
            return self._create_date_windows(events)
        elif self.window_type == "count":
            return self._create_count_windows(events)
        elif self.window_type == "incremental":
            return self._create_incremental_windows(events)
        else:
            raise ValueError(f"Unsupported window type: {self.window_type}")
    
    def _create_date_windows(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create date-based sliding windows.
        
        Compatible with Task-1 and Task-3 date-based windowing.
        """
        # Sort events by event_date
        sorted_events = sorted(events, key=lambda x: datetime.fromisoformat(x['event_date'].replace('Z', '+00:00')))
        
        if not sorted_events:
            return []
        
        # Get date range  
        start_date = datetime.fromisoformat(sorted_events[0]['event_date'].replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(sorted_events[-1]['event_date'].replace('Z', '+00:00'))
        
        windows = []
        current_start = start_date
        window_index = 0
        
        while current_start <= end_date:
            window_end = current_start + timedelta(days=self.window_size)
            
            # Filter events in this window
            window_events = [
                event for event in sorted_events 
                if current_start <= datetime.fromisoformat(event['event_date'].replace('Z', '+00:00')) < window_end
            ]
            
            if window_events:  # Only create window if it has events
                window = {
                    'window_index': window_index,
                    'start_date': current_start.isoformat(),
                    'end_date': window_end.isoformat(),
                    'events': window_events,
                    'event_count': len(window_events)
                }
                windows.append(window)
                window_index += 1
            
            # Move to next window
            current_start += timedelta(days=self.stride)
        
        logging.info(f"Created {len(windows)} date-based windows")
        return windows
    
    def _create_count_windows(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create count-based windows.
        
        Compatible with Task-2 count-based windowing.
        """
        if not events:
            return []
        
        # Group events by topic first (for Task-2 compatibility)
        topic_events = defaultdict(list)
        for event in events:
            topic = event.get('topic', 'default')
            topic_events[topic].append(event)
        
        # Create windows for each topic
        all_windows = []
        window_index = 0
        
        # Process events in chronological order across all topics
        all_events_sorted = sorted(events, key=lambda x: datetime.fromisoformat(x['date'].replace('Z', '+00:00')))
        
        for i in range(0, len(all_events_sorted) - int(self.window_size) + 1, int(self.stride)):
            window_events = all_events_sorted[i:i + int(self.window_size)]
            
            if window_events:
                window = {
                    'window_index': window_index,
                    'start_date': window_events[0]['date'],
                    'end_date': window_events[-1]['date'],
                    'events': window_events,
                    'event_count': len(window_events)
                }
                all_windows.append(window)
                window_index += 1
        
        logging.info(f"Created {len(all_windows)} count-based windows")
        return all_windows
    
    def _create_incremental_windows(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create incremental windows for streaming processing.
        
        Compatible with Task-1 incremental clustering.
        Each window contains events up to that point (cumulative).
        """
        # Sort events by date
        sorted_events = sorted(events, key=lambda x: datetime.fromisoformat(x['date'].replace('Z', '+00:00')))
        
        windows = []
        for i, event in enumerate(sorted_events):
            # Incremental window contains all events up to current index
            window_events = sorted_events[:i+1]
            
            window = {
                'window_index': i,
                'start_date': sorted_events[0]['date'],
                'end_date': event['date'],
                'events': window_events,
                'event_count': len(window_events),
                'new_event': event  # Current event being processed
            }
            windows.append(window)
        
        logging.info(f"Created {len(windows)} incremental windows")
        return windows
    
    def get_overlapping_windows(self, target_date: str, windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get all windows that contain the target date.
        
        Useful for Task-3 when finding relevant windows for QA pairs.
        """
        target_dt = datetime.fromisoformat(target_date.replace('Z', '+00:00'))
        
        matching_windows = []
        for window in windows:
            start_dt = datetime.fromisoformat(window['start_date'].replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(window['end_date'].replace('Z', '+00:00'))
            
            if start_dt <= target_dt < end_dt:
                matching_windows.append(window)
        
        return matching_windows
    
    def filter_events_by_date_range(self, events: List[Dict[str, Any]], 
                                   start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Filter events within a specific date range.
        
        Args:
            events: List of events
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            
        Returns:
            Filtered events
        """
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        filtered = [
            event for event in events
            if start_dt <= datetime.fromisoformat(event['date'].replace('Z', '+00:00')) < end_dt
        ]
        
        return filtered
    
    def get_window_summary(self, windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics of created windows.
        
        Returns:
            Dictionary with window statistics
        """
        if not windows:
            return {'total_windows': 0, 'total_events': 0, 'avg_events_per_window': 0}
        
        total_events = sum(w['event_count'] for w in windows)
        avg_events = total_events / len(windows)
        
        return {
            'total_windows': len(windows),
            'total_events': total_events,
            'avg_events_per_window': avg_events,
            'window_type': self.window_type,
            'window_size': self.window_size,
            'stride': self.stride
        }
