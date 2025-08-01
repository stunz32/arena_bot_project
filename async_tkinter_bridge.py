#!/usr/bin/env python3
"""
Async-Tkinter Bridge for S-Tier Logging Integration

Provides hybrid event loop integration allowing Tkinter GUI to coexist
with asyncio-based S-Tier logging system without blocking operations.

Architecture:
- Main thread runs asyncio event loop
- Tkinter runs in dedicated thread with periodic async coordination
- S-Tier logging operates in async context
- Thread-safe communication via queues and callbacks

Performance Targets:
- GUI responsiveness: <16ms frame time (60fps)
- Async operation latency: <50μs
- Memory overhead: <5MB
- CPU overhead: <0.5%
"""

import asyncio
import threading
import tkinter as tk
from typing import Any, Callable, Optional, Dict, List
import queue
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import weakref

# Import S-Tier logging system
try:
    from arena_bot.logging_system import get_s_tier_logger, setup_s_tier_logging
    STIER_AVAILABLE = True
except ImportError:
    STIER_AVAILABLE = False


class AsyncTkinterBridge:
    """
    High-performance bridge between asyncio and Tkinter with S-Tier logging integration.
    
    Provides thread-safe communication and coordination between async operations
    and synchronous GUI updates with minimal overhead.
    """
    
    def __init__(self, logger_name: str = "async_tkinter_bridge"):
        """
        Initialize async-Tkinter bridge.
        
        Args:
            logger_name: Name for S-Tier logger instance
        """
        self.logger_name = logger_name
        self.logger: Optional[Any] = None
        
        # Thread management
        self.gui_thread: Optional[threading.Thread] = None
        self.gui_thread_id: Optional[int] = None
        self.main_loop: Optional[asyncio.AbstractEventLoop] = None
        self.shutdown_event = asyncio.Event()
        
        # Communication channels
        self.gui_to_async_queue = queue.Queue()  # Thread-safe queue for GUI -> async
        self.async_to_gui_queue = asyncio.Queue()  # Async queue for async -> GUI
        
        # Tkinter integration
        self.root: Optional[tk.Tk] = None
        self.gui_callbacks: Dict[str, List[Callable]] = {}
        self.periodic_tasks: List[asyncio.Task] = []
        
        # Performance monitoring
        self.performance_stats = {
            'gui_updates_per_second': 0,
            'async_operations_per_second': 0,
            'average_bridge_latency_us': 0,
            'memory_usage_mb': 0
        }
        self.stats_lock = threading.Lock()
        
        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tkinter_bridge")
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the bridge with S-Tier logging.
        
        Args:
            config: Optional configuration for S-Tier logging
        """
        # Initialize S-Tier logging if available
        if STIER_AVAILABLE:
            self.logger = await get_s_tier_logger(self.logger_name)
            await self.logger.info("🌉 AsyncTkinterBridge initializing", extra={
                'bridge_config': {
                    'stier_available': True,
                    'thread_pool_workers': 2,
                    'performance_monitoring': True
                }
            })
        else:
            # Fallback to standard logging
            self.logger = logging.getLogger(self.logger_name)
            self.logger.info("AsyncTkinterBridge initializing (S-Tier unavailable)")
        
        # Store reference to main event loop
        self.main_loop = asyncio.get_event_loop()
        
        # Start performance monitoring
        self._start_performance_monitoring()
    
    def create_root(self, **tkinter_kwargs) -> tk.Tk:
        """
        Create Tkinter root window in GUI thread.
        
        Args:
            **tkinter_kwargs: Arguments passed to tk.Tk()
            
        Returns:
            Tkinter root window
        """
        if self.root is not None:
            raise RuntimeError("Root window already created")
        
        # Create root in current thread (will be GUI thread)
        self.root = tk.Tk(**tkinter_kwargs)
        self.gui_thread_id = threading.get_ident()
        
        # Setup periodic async coordination
        self._setup_periodic_gui_updates()
        
        return self.root
    
    def _setup_periodic_gui_updates(self) -> None:
        """Setup periodic updates to coordinate async operations with GUI."""
        def process_async_messages():
            """Process messages from async context in GUI thread."""
            try:
                # Non-blocking check for async messages
                while True:
                    try:
                        # Get message without blocking
                        message = self.gui_to_async_queue.get_nowait()
                        self._handle_gui_message(message)
                    except queue.Empty:
                        break
                
                # Schedule next update
                if self.root and not self.shutdown_event.is_set():
                    self.root.after(16, process_async_messages)  # ~60fps
                    
            except Exception as e:
                if self.logger:
                    if hasattr(self.logger, 'error'):
                        self.logger.error(f"GUI message processing error: {e}")
                
        # Start periodic processing
        if self.root:
            self.root.after(16, process_async_messages)
    
    def _handle_gui_message(self, message: Dict[str, Any]) -> None:
        """Handle message from async context in GUI thread."""
        try:
            message_type = message.get('type')
            
            if message_type == 'callback':
                callback = message.get('callback')
                args = message.get('args', ())
                kwargs = message.get('kwargs', {})
                
                if callback:
                    callback(*args, **kwargs)
            
            elif message_type == 'widget_update':
                widget = message.get('widget')
                method = message.get('method')
                args = message.get('args', ())
                kwargs = message.get('kwargs', {})
                
                if widget and method and hasattr(widget, method):
                    getattr(widget, method)(*args, **kwargs)
            
            # Update performance stats
            with self.stats_lock:
                self.performance_stats['gui_updates_per_second'] += 1
                
        except Exception as e:
            if self.logger:
                if hasattr(self.logger, 'error'):
                    self.logger.error(f"GUI message handling error: {e}")
    
    async def schedule_gui_callback(self, callback: Callable, *args, **kwargs) -> None:
        """
        Schedule callback to run in GUI thread from async context.
        
        Args:
            callback: Function to call in GUI thread
            *args: Positional arguments for callback
            **kwargs: Keyword arguments for callback
        """
        message = {
            'type': 'callback',
            'callback': callback,
            'args': args,
            'kwargs': kwargs,
            'timestamp': time.perf_counter()
        }
        
        # Send to GUI thread via thread-safe queue
        self.gui_to_async_queue.put(message)
        
        # Update performance stats
        if STIER_AVAILABLE and self.logger:
            await self.logger.debug("GUI callback scheduled", extra={
                'callback_name': callback.__name__ if hasattr(callback, '__name__') else str(callback),
                'args_count': len(args),
                'kwargs_count': len(kwargs)
            })
    
    async def update_widget_async(self, widget: tk.Widget, method: str, *args, **kwargs) -> None:
        """
        Update Tkinter widget from async context.
        
        Args:
            widget: Tkinter widget to update
            method: Method name to call on widget
            *args: Positional arguments for method
            **kwargs: Keyword arguments for method
        """
        message = {
            'type': 'widget_update',
            'widget': widget,
            'method': method,
            'args': args,
            'kwargs': kwargs,
            'timestamp': time.perf_counter()
        }
        
        self.gui_to_async_queue.put(message)
    
    def run_gui_thread(self, root_factory: Callable[[], tk.Tk]) -> None:
        """
        Run Tkinter GUI in dedicated thread.
        
        Args:
            root_factory: Function that creates and configures Tkinter root
        """
        def gui_thread_main():
            """Main function for GUI thread."""
            try:
                # Create root window
                self.root = root_factory()
                self.gui_thread_id = threading.get_ident()
                
                # Setup coordination
                self._setup_periodic_gui_updates()
                
                # Run Tkinter main loop
                self.root.mainloop()
                
            except Exception as e:
                if self.logger:
                    if hasattr(self.logger, 'error'):
                        self.logger.error(f"GUI thread error: {e}")
                
                # Signal shutdown
                if self.main_loop:
                    self.main_loop.call_soon_threadsafe(self.shutdown_event.set)
        
        # Start GUI thread
        self.gui_thread = threading.Thread(
            target=gui_thread_main,
            name="TkinterGUI",
            daemon=True
        )
        self.gui_thread.start()
    
    async def run_async_main_loop(self, async_main: Callable) -> None:
        """
        Run main async application logic.
        
        Args:
            async_main: Async function containing main application logic
        """
        try:
            if STIER_AVAILABLE and self.logger:
                await self.logger.info("🚀 Starting async main loop", extra={
                    'main_function': async_main.__name__ if hasattr(async_main, '__name__') else str(async_main)
                })
            
            # Run main async application
            await async_main()
            
        except Exception as e:
            if STIER_AVAILABLE and self.logger:
                await self.logger.error("💥 Async main loop error", extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }, exc_info=True)
            raise
        finally:
            # Signal shutdown
            self.shutdown_event.set()
    
    def _start_performance_monitoring(self) -> None:
        """Start background performance monitoring."""
        async def monitor_performance():
            """Monitor bridge performance metrics."""
            while not self.shutdown_event.is_set():
                try:
                    # Collect performance metrics
                    with self.stats_lock:
                        current_stats = self.performance_stats.copy()
                    
                    if STIER_AVAILABLE and self.logger:
                        await self.logger.debug("🔍 Bridge performance metrics", extra={
                            'performance': current_stats,
                            'gui_thread_alive': self.gui_thread.is_alive() if self.gui_thread else False,
                            'active_tasks': len(self.periodic_tasks)
                        })
                    
                    # Reset counters
                    with self.stats_lock:
                        self.performance_stats['gui_updates_per_second'] = 0
                        self.performance_stats['async_operations_per_second'] = 0
                    
                    # Wait before next monitoring cycle
                    await asyncio.sleep(10)  # Monitor every 10 seconds
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if STIER_AVAILABLE and self.logger:
                        await self.logger.warning(f"Performance monitoring error: {e}")
        
        # Start monitoring task
        if self.main_loop:
            task = self.main_loop.create_task(monitor_performance())
            self.periodic_tasks.append(task)
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the bridge."""
        if STIER_AVAILABLE and self.logger:
            await self.logger.info("🛑 Shutting down AsyncTkinterBridge")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Cancel periodic tasks
        for task in self.periodic_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Shutdown GUI if running
        if self.root:
            await self.schedule_gui_callback(self.root.quit)
        
        # Wait for GUI thread to finish
        if self.gui_thread and self.gui_thread.is_alive():
            self.gui_thread.join(timeout=5.0)
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        if STIER_AVAILABLE and self.logger:
            await self.logger.info("✅ AsyncTkinterBridge shutdown complete")


@asynccontextmanager
async def async_tkinter_app(root_factory: Callable[[], tk.Tk], 
                           logger_name: str = "arena_bot_gui"):
    """
    Context manager for async Tkinter applications with S-Tier logging.
    
    Args:
        root_factory: Function that creates and configures Tkinter root
        logger_name: Name for S-Tier logger
        
    Example:
        async def create_gui():
            root = tk.Tk()
            root.title("Arena Bot")
            return root
            
        async def main():
            async with async_tkinter_app(create_gui) as bridge:
                # Your async application logic here
                await some_async_operation()
    """
    bridge = AsyncTkinterBridge(logger_name)
    
    try:
        # Initialize bridge
        await bridge.initialize()
        
        # Start GUI thread
        bridge.run_gui_thread(root_factory)
        
        # Wait a moment for GUI to initialize
        await asyncio.sleep(0.1)
        
        yield bridge
        
    finally:
        # Cleanup
        await bridge.shutdown()


# Utility functions for common operations
async def call_in_gui_thread(bridge: AsyncTkinterBridge, 
                            callback: Callable, 
                            *args, **kwargs) -> None:
    """Convenience function to call function in GUI thread."""
    await bridge.schedule_gui_callback(callback, *args, **kwargs)


async def update_widget(bridge: AsyncTkinterBridge, 
                       widget: tk.Widget, 
                       method: str, 
                       *args, **kwargs) -> None:
    """Convenience function to update widget from async context."""
    await bridge.update_widget_async(widget, method, *args, **kwargs)


# Module exports
__all__ = [
    'AsyncTkinterBridge',
    'async_tkinter_app',
    'call_in_gui_thread',
    'update_widget'
]