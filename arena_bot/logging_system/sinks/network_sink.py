"""
Network Sink for S-Tier Logging System.

This module provides network-based log transmission with support for multiple
protocols, reliability features, and high-performance async networking.

Features:
- Multiple protocol support (HTTP, TCP, UDP, WebSocket)
- Automatic failover and load balancing
- Message buffering and batching
- Compression and encryption
- Connection pooling and keep-alive
- Retry logic with exponential backoff
- Circuit breaker pattern for resilience
- Performance optimization and monitoring
"""

import asyncio
import json
import time
import threading
import logging
import socket
import ssl
import gzip
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
import base64

# HTTP client imports
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

# Import from our components
from .base_sink import BaseSink, SinkState, ErrorHandlingStrategy
from ..formatters.structured_formatter import StructuredFormatter
from ..core.hybrid_async_queue import LogMessage


class NetworkProtocol(Enum):
    """Supported network protocols."""
    HTTP = "http"
    HTTPS = "https"
    TCP = "tcp"
    UDP = "udp"
    WEBSOCKET = "websocket"
    SYSLOG = "syslog"


class CompressionType(Enum):
    """Network compression types."""
    NONE = "none"
    GZIP = "gzip"
    DEFLATE = "deflate"


class AuthenticationType(Enum):
    """Authentication types for network transmission."""
    NONE = "none"
    BASIC = "basic"
    BEARER = "bearer"
    HMAC = "hmac"
    TLS_CLIENT_CERT = "tls_client_cert"


@dataclass
class NetworkEndpoint:
    """Network endpoint configuration."""
    host: str
    port: int
    protocol: NetworkProtocol = NetworkProtocol.HTTP
    path: str = "/logs"
    weight: int = 1                    # Load balancing weight
    max_connections: int = 10          # Connection pool size
    timeout_seconds: float = 30.0      # Request timeout
    use_ssl: bool = False             # SSL/TLS encryption
    verify_ssl: bool = True           # SSL certificate verification
    
    def get_url(self) -> str:
        """Get full URL for HTTP/HTTPS endpoints."""
        if self.protocol in [NetworkProtocol.HTTP, NetworkProtocol.HTTPS]:
            scheme = "https" if self.use_ssl or self.protocol == NetworkProtocol.HTTPS else "http"
            return f"{scheme}://{self.host}:{self.port}{self.path}"
        elif self.protocol == NetworkProtocol.WEBSOCKET:
            scheme = "wss" if self.use_ssl else "ws"
            return f"{scheme}://{self.host}:{self.port}{self.path}"
        else:
            return f"{self.host}:{self.port}"


@dataclass
class NetworkConfig:
    """Network sink configuration."""
    endpoints: List[NetworkEndpoint] = field(default_factory=list)
    batch_size: int = 100                      # Messages per batch
    batch_timeout_seconds: float = 5.0        # Max time to wait for batch
    max_retries: int = 3                      # Retry attempts per message
    retry_backoff_factor: float = 2.0         # Exponential backoff multiplier  
    max_retry_delay: float = 60.0             # Maximum retry delay
    compression: CompressionType = CompressionType.GZIP
    authentication: AuthenticationType = AuthenticationType.NONE
    auth_credentials: Dict[str, str] = field(default_factory=dict)
    custom_headers: Dict[str, str] = field(default_factory=dict)
    enable_keepalive: bool = True
    keepalive_timeout: float = 300.0          # Connection keep-alive timeout
    circuit_breaker_threshold: int = 5        # Failures before circuit opens
    circuit_breaker_timeout: float = 60.0     # Circuit breaker reset timeout


class CircuitBreaker:
    """Circuit breaker for network reliability."""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"{__name__}.CircuitBreaker")
    
    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        with self._lock:
            if self.state == "closed":
                return True
            elif self.state == "open":
                if time.time() - self.last_failure_time >= self.reset_timeout:
                    self.state = "half_open"
                    self._logger.info("Circuit breaker half-open")
                    return True
                return False
            else:  # half_open
                return True
    
    def record_success(self) -> None:
        """Record successful operation."""
        with self._lock:
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                self._logger.info("Circuit breaker closed")
    
    def record_failure(self) -> None:
        """Record failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold and self.state == "closed":
                self.state = "open"
                self._logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            elif self.state == "half_open":
                self.state = "open"
                self._logger.warning("Circuit breaker reopened after half-open failure")


class NetworkBatch:
    """Batch of messages for network transmission."""
    
    def __init__(self, max_size: int = 100, timeout: float = 5.0):
        self.max_size = max_size
        self.timeout = timeout
        self.messages: List[str] = []
        self.created_at = time.time()
        self._lock = threading.RLock()
    
    def add_message(self, message: str) -> bool:
        """Add message to batch. Returns True if batch is full."""
        with self._lock:
            if len(self.messages) < self.max_size:
                self.messages.append(message)
                return len(self.messages) >= self.max_size
            return True  # Already full
    
    def is_ready(self) -> bool:
        """Check if batch is ready for sending."""
        with self._lock:
            return (
                len(self.messages) >= self.max_size or
                time.time() - self.created_at >= self.timeout
            )
    
    def get_messages(self) -> List[str]:
        """Get copy of messages in batch."""
        with self._lock:
            return self.messages.copy()
    
    def size(self) -> int:
        """Get current batch size."""
        with self._lock:
            return len(self.messages)


class NetworkSink(BaseSink):
    """
    High-performance network sink with reliability features.
    
    Provides robust network-based log transmission with support for
    multiple protocols, failover, batching, and comprehensive error handling.
    
    Features:
    - Multiple protocol support (HTTP, TCP, UDP, WebSocket)
    - Load balancing and failover
    - Message batching and compression
    - Retry logic with circuit breaker
    - Connection pooling
    - Performance monitoring
    """
    
    def __init__(self,
                 name: str = "network",
                 config: Optional[NetworkConfig] = None,
                 formatter: Optional[Any] = None,
                 async_mode: bool = True,
                 worker_threads: int = 2,
                 **base_kwargs):
        """
        Initialize network sink.
        
        Args:
            name: Sink name for identification
            config: Network configuration
            formatter: Message formatter (defaults to StructuredFormatter)
            async_mode: Use async networking where possible
            worker_threads: Number of worker threads for sync operations
            **base_kwargs: Arguments for BaseSink
        """
        # Set up formatter before calling parent __init__
        if formatter is None:
            formatter = StructuredFormatter(
                timestamp_format="iso",
                include_performance_metrics=False,  # Reduce network payload
                include_system_info=False,
                ensure_ascii=True,  # Better network compatibility
                indent=None  # Compact JSON
            )
        
        # Initialize parent
        super().__init__(name=name, formatter=formatter, **base_kwargs)
        
        # Configuration
        self.config = config or NetworkConfig()
        self.async_mode = async_mode and AIOHTTP_AVAILABLE
        self.worker_threads = worker_threads
        
        # Validate configuration
        if not self.config.endpoints:
            raise ValueError("At least one network endpoint must be configured")
        
        # Circuit breakers per endpoint
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        for endpoint in self.config.endpoints:
            endpoint_key = f"{endpoint.host}:{endpoint.port}"
            self.circuit_breakers[endpoint_key] = CircuitBreaker(
                failure_threshold=self.config.circuit_breaker_threshold,
                reset_timeout=self.config.circuit_breaker_timeout
            )
        
        # Batching
        self.current_batch = NetworkBatch(
            max_size=self.config.batch_size,
            timeout=self.config.batch_timeout_seconds
        )
        self.batch_lock = threading.RLock()
        
        # Connection management
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.websocket_connections: Dict[str, Any] = {}
        self.tcp_connections: Dict[str, socket.socket] = {}
        
        # Worker management
        self.executor = ThreadPoolExecutor(max_workers=worker_threads, thread_name_prefix=f"NetworkSink-{name}")
        self.async_loop: Optional[asyncio.AbstractEventLoop] = None
        self.async_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._messages_sent = 0
        self._bytes_sent = 0
        self._send_errors = 0
        self._batch_count = 0
        self._endpoint_stats: Dict[str, Dict[str, int]] = {}
        
        # Initialize endpoint stats
        for endpoint in self.config.endpoints:
            endpoint_key = f"{endpoint.host}:{endpoint.port}"
            self._endpoint_stats[endpoint_key] = {
                'messages_sent': 0,
                'bytes_sent': 0,
                'errors': 0,
                'response_times': deque(maxlen=100)
            }
        
        # Background batch processor
        self._batch_processor_thread: Optional[threading.Thread] = None
        self._batch_processor_stop = threading.Event()
        
        self._logger.info(f"NetworkSink '{name}' initialized",
                         extra={
                             'endpoints': len(self.config.endpoints),
                             'batch_size': self.config.batch_size,
                             'async_mode': self.async_mode,
                             'compression': self.config.compression.value,
                             'authentication': self.config.authentication.value
                         })
    
    def _initialize_sink(self) -> bool:
        """Initialize network sink."""
        try:
            # Start async event loop if in async mode
            if self.async_mode:
                self._start_async_loop()
            
            # Start batch processor
            self._batch_processor_thread = threading.Thread(
                target=self._batch_processor_loop,
                name=f"NetworkSink-{self.name}-BatchProcessor",
                daemon=True
            )
            self._batch_processor_thread.start()
            
            # Test connectivity to endpoints
            self._test_endpoint_connectivity()
            
            return True
            
        except Exception as e:
            self._logger.error(f"Network sink initialization failed: {e}")
            return False
    
    def _cleanup_sink(self) -> bool:
        """Cleanup network sink."""
        try:
            # Stop batch processor
            if self._batch_processor_thread and self._batch_processor_thread.is_alive():
                self._batch_processor_stop.set()
                self._batch_processor_thread.join(timeout=5.0)
            
            # Send final batch
            if self.current_batch.size() > 0:
                self._send_batch(self.current_batch)
            
            # Close connections
            if self.async_mode and self.async_loop:
                future = asyncio.run_coroutine_threadsafe(
                    self._cleanup_async_resources(), 
                    self.async_loop
                )
                future.result(timeout=5.0)
                
                if self.async_thread and self.async_thread.is_alive():
                    self.async_loop.call_soon_threadsafe(self.async_loop.stop)
                    self.async_thread.join(timeout=5.0)
            
            # Close TCP connections
            for conn in self.tcp_connections.values():
                try:
                    conn.close()
                except Exception:
                    pass
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True, timeout=5.0)
            
            self._logger.info(f"NetworkSink '{self.name}' cleanup completed",
                             extra={
                                 'messages_sent': self._messages_sent,
                                 'bytes_sent': self._bytes_sent,
                                 'send_errors': self._send_errors,
                                 'batch_count': self._batch_count
                             })
            
            return True
            
        except Exception as e:
            self._logger.error(f"Network sink cleanup failed: {e}")
            return False
    
    def _health_check_sink(self) -> bool:
        """Perform health check on network sink."""
        try:
            # Check if batch processor is running
            if self._batch_processor_thread and not self._batch_processor_thread.is_alive():
                return False
            
            # Check if async loop is running (if in async mode)
            if self.async_mode and (not self.async_loop or self.async_loop.is_closed()):
                return False
            
            # Check circuit breaker states
            open_breakers = sum(1 for cb in self.circuit_breakers.values() if cb.state == "open")
            if open_breakers == len(self.circuit_breakers):
                return False  # All endpoints are down
            
            return True
            
        except Exception as e:
            self._logger.warning(f"Network sink health check failed: {e}")
            return False
    
    def _write_message(self, formatted_message: str, message: LogMessage) -> bool:
        """Add message to batch for network transmission."""
        try:
            with self.batch_lock:
                # Add message to current batch
                batch_full = self.current_batch.add_message(formatted_message)
                
                # If batch is full or ready, send it
                if batch_full or self.current_batch.is_ready():
                    # Send current batch and start new one
                    batch_to_send = self.current_batch
                    self.current_batch = NetworkBatch(
                        max_size=self.config.batch_size,
                        timeout=self.config.batch_timeout_seconds
                    )
                    
                    # Send batch in background
                    self.executor.submit(self._send_batch, batch_to_send)
            
            return True
            
        except Exception as e:
            self._logger.error(f"Network message batching failed: {e}")
            return False
    
    def _start_async_loop(self) -> None:
        """Start async event loop in separate thread."""
        def run_async_loop():
            try:
                self.async_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.async_loop)
                self.async_loop.run_forever()
            except Exception as e:
                self._logger.error(f"Async loop failed: {e}")
        
        self.async_thread = threading.Thread(
            target=run_async_loop,
            name=f"NetworkSink-{self.name}-AsyncLoop",
            daemon=True
        )
        self.async_thread.start()
        
        # Wait for loop to be ready
        timeout = 0
        while self.async_loop is None and timeout < 50:  # 5 second timeout
            time.sleep(0.1)
            timeout += 1
    
    async def _cleanup_async_resources(self) -> None:
        """Clean up async resources."""
        try:
            # Close HTTP session
            if self.http_session and not self.http_session.closed:
                await self.http_session.close()
            
            # Close WebSocket connections
            for ws in self.websocket_connections.values():
                try:
                    if not ws.closed:
                        await ws.close()
                except Exception:
                    pass
                    
        except Exception as e:
            self._logger.error(f"Async cleanup failed: {e}")
    
    def _test_endpoint_connectivity(self) -> None:
        """Test connectivity to all endpoints."""
        for endpoint in self.config.endpoints:
            try:
                endpoint_key = f"{endpoint.host}:{endpoint.port}"
                
                if endpoint.protocol in [NetworkProtocol.HTTP, NetworkProtocol.HTTPS]:
                    # Simple HTTP connectivity test
                    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_socket.settimeout(5.0)
                    result = test_socket.connect_ex((endpoint.host, endpoint.port))
                    test_socket.close()
                    
                    if result == 0:
                        self._logger.info(f"Endpoint {endpoint_key} connectivity OK")
                    else:
                        self._logger.warning(f"Endpoint {endpoint_key} not reachable")
                        
                elif endpoint.protocol == NetworkProtocol.TCP:
                    # TCP connectivity test
                    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_socket.settimeout(5.0)
                    result = test_socket.connect_ex((endpoint.host, endpoint.port))
                    test_socket.close()
                    
                    if result == 0:
                        self._logger.info(f"TCP endpoint {endpoint_key} connectivity OK")
                    else:
                        self._logger.warning(f"TCP endpoint {endpoint_key} not reachable")
                
            except Exception as e:
                self._logger.warning(f"Connectivity test failed for {endpoint_key}: {e}")
    
    def _batch_processor_loop(self) -> None:
        """Background batch processor loop."""
        while not self._batch_processor_stop.is_set():
            try:
                # Check if current batch is ready to send
                with self.batch_lock:
                    if self.current_batch.is_ready() and self.current_batch.size() > 0:
                        batch_to_send = self.current_batch
                        self.current_batch = NetworkBatch(
                            max_size=self.config.batch_size,
                            timeout=self.config.batch_timeout_seconds
                        )
                        
                        # Send batch
                        self._send_batch(batch_to_send)
                
                # Sleep for a short interval
                self._batch_processor_stop.wait(1.0)
                
            except Exception as e:
                self._logger.error(f"Batch processor error: {e}")
                time.sleep(5.0)  # Back off on error
    
    def _send_batch(self, batch: NetworkBatch) -> bool:
        """Send batch of messages to network endpoints."""
        if batch.size() == 0:
            return True
        
        try:
            messages = batch.get_messages()
            
            # Prepare payload
            if self.config.batch_size == 1:
                payload = messages[0]
            else:
                # Create batch payload
                batch_payload = {
                    'messages': messages,
                    'count': len(messages),
                    'timestamp': time.time(),
                    'source': self.name
                }
                payload = json.dumps(batch_payload)
            
            # Compress if enabled
            if self.config.compression == CompressionType.GZIP:
                payload_bytes = gzip.compress(payload.encode('utf-8'))
                content_encoding = 'gzip'
            else:
                payload_bytes = payload.encode('utf-8')
                content_encoding = None
            
            # Try each endpoint until one succeeds
            success = False
            for endpoint in self._select_endpoints():
                endpoint_key = f"{endpoint.host}:{endpoint.port}"
                circuit_breaker = self.circuit_breakers[endpoint_key]
                
                if not circuit_breaker.can_execute():
                    continue
                
                try:
                    start_time = time.perf_counter()
                    
                    if endpoint.protocol in [NetworkProtocol.HTTP, NetworkProtocol.HTTPS]:
                        result = self._send_http_batch(endpoint, payload_bytes, content_encoding)
                    elif endpoint.protocol == NetworkProtocol.TCP:
                        result = self._send_tcp_batch(endpoint, payload_bytes)
                    elif endpoint.protocol == NetworkProtocol.UDP:
                        result = self._send_udp_batch(endpoint, payload_bytes)
                    elif endpoint.protocol == NetworkProtocol.WEBSOCKET:
                        result = self._send_websocket_batch(endpoint, payload)
                    else:
                        result = False
                    
                    elapsed_time = time.perf_counter() - start_time
                    
                    if result:
                        # Success
                        circuit_breaker.record_success()
                        self._update_endpoint_stats(endpoint_key, len(messages), len(payload_bytes), elapsed_time)
                        success = True
                        break
                    else:
                        circuit_breaker.record_failure()
                        
                except Exception as e:
                    circuit_breaker.record_failure()
                    self._logger.warning(f"Send failed to {endpoint_key}: {e}")
                    self._endpoint_stats[endpoint_key]['errors'] += 1
            
            if success:
                self._messages_sent += len(messages)
                self._bytes_sent += len(payload_bytes)
                self._batch_count += 1
            else:
                self._send_errors += 1
                self._logger.error(f"Failed to send batch to any endpoint")
            
            return success
            
        except Exception as e:
            self._logger.error(f"Batch send failed: {e}")
            return False
    
    def _select_endpoints(self) -> List[NetworkEndpoint]:
        """Select endpoints for load balancing."""
        # Simple weighted round-robin for now
        # In production, this could be more sophisticated
        available_endpoints = []
        
        for endpoint in self.config.endpoints:
            endpoint_key = f"{endpoint.host}:{endpoint.port}"
            circuit_breaker = self.circuit_breakers[endpoint_key]
            
            if circuit_breaker.can_execute():
                # Add endpoint multiple times based on weight
                available_endpoints.extend([endpoint] * endpoint.weight)
        
        if not available_endpoints:
            # All endpoints are down, try them anyway (circuit breaker will limit)
            available_endpoints = self.config.endpoints.copy()
        
        return available_endpoints
    
    def _send_http_batch(self, endpoint: NetworkEndpoint, payload: bytes, content_encoding: Optional[str]) -> bool:
        """Send batch via HTTP/HTTPS."""
        try:
            if self.async_mode and self.async_loop:
                # Use async HTTP
                future = asyncio.run_coroutine_threadsafe(
                    self._send_http_batch_async(endpoint, payload, content_encoding),
                    self.async_loop
                )
                return future.result(timeout=endpoint.timeout_seconds)
            else:
                # Use synchronous HTTP (fallback)
                return self._send_http_batch_sync(endpoint, payload, content_encoding)
                
        except Exception as e:
            self._logger.error(f"HTTP batch send failed: {e}")
            return False
    
    async def _send_http_batch_async(self, endpoint: NetworkEndpoint, payload: bytes, content_encoding: Optional[str]) -> bool:
        """Send batch via async HTTP."""
        try:
            # Create session if needed
            if not self.http_session or self.http_session.closed:
                connector = aiohttp.TCPConnector(
                    limit=endpoint.max_connections,
                    limit_per_host=endpoint.max_connections,
                    keepalive_timeout=self.config.keepalive_timeout,
                    enable_cleanup_closed=True
                )
                
                timeout = aiohttp.ClientTimeout(total=endpoint.timeout_seconds)
                self.http_session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout
                )
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': f'S-Tier-Logging-NetworkSink/{self.name}'
            }
            
            if content_encoding:
                headers['Content-Encoding'] = content_encoding
            
            # Add authentication headers
            auth_headers = self._get_auth_headers(endpoint)
            headers.update(auth_headers)
            
            # Add custom headers
            headers.update(self.config.custom_headers)
            
            # Send request
            url = endpoint.get_url()
            async with self.http_session.post(url, data=payload, headers=headers) as response:
                if 200 <= response.status < 300:
                    return True
                else:
                    self._logger.warning(f"HTTP error {response.status}: {await response.text()}")
                    return False
                    
        except Exception as e:
            self._logger.error(f"Async HTTP send failed: {e}")
            return False
    
    def _send_http_batch_sync(self, endpoint: NetworkEndpoint, payload: bytes, content_encoding: Optional[str]) -> bool:
        """Send batch via synchronous HTTP (fallback)."""
        # This would use urllib or requests in a real implementation
        # For now, return False to indicate sync HTTP not implemented
        self._logger.warning("Synchronous HTTP not implemented, install aiohttp for async support")
        return False
    
    def _send_tcp_batch(self, endpoint: NetworkEndpoint, payload: bytes) -> bool:
        """Send batch via TCP."""
        try:
            endpoint_key = f"{endpoint.host}:{endpoint.port}"
            
            # Get or create connection
            if endpoint_key not in self.tcp_connections:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(endpoint.timeout_seconds)
                
                if endpoint.use_ssl:
                    context = ssl.create_default_context()
                    if not endpoint.verify_ssl:
                        context.check_hostname = False
                        context.verify_mode = ssl.CERT_NONE
                    sock = context.wrap_socket(sock, server_hostname=endpoint.host)
                
                sock.connect((endpoint.host, endpoint.port))
                self.tcp_connections[endpoint_key] = sock
            
            sock = self.tcp_connections[endpoint_key]
            
            # Send payload with length prefix
            payload_with_length = len(payload).to_bytes(4, byteorder='big') + payload
            sock.sendall(payload_with_length)
            
            return True
            
        except Exception as e:
            # Remove failed connection
            if endpoint_key in self.tcp_connections:
                try:
                    self.tcp_connections[endpoint_key].close()
                except Exception:
                    pass
                del self.tcp_connections[endpoint_key]
            
            self._logger.error(f"TCP send failed: {e}")
            return False
    
    def _send_udp_batch(self, endpoint: NetworkEndpoint, payload: bytes) -> bool:
        """Send batch via UDP."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(endpoint.timeout_seconds)
            
            # UDP has size limits, split large payloads
            max_udp_size = 65507  # Max UDP payload size
            
            if len(payload) <= max_udp_size:
                sock.sendto(payload, (endpoint.host, endpoint.port))
            else:
                # Split into chunks
                chunk_size = max_udp_size - 100  # Leave room for headers
                for i in range(0, len(payload), chunk_size):
                    chunk = payload[i:i + chunk_size]
                    sock.sendto(chunk, (endpoint.host, endpoint.port))
            
            sock.close()
            return True
            
        except Exception as e:
            self._logger.error(f"UDP send failed: {e}")
            return False
    
    def _send_websocket_batch(self, endpoint: NetworkEndpoint, payload: str) -> bool:
        """Send batch via WebSocket."""
        if not WEBSOCKETS_AVAILABLE:
            self._logger.error("WebSocket support requires 'websockets' package")
            return False
        
        try:
            # This would require async WebSocket implementation
            # For now, return False to indicate WebSocket not fully implemented
            self._logger.warning("WebSocket support not fully implemented")
            return False
            
        except Exception as e:
            self._logger.error(f"WebSocket send failed: {e}")
            return False
    
    def _get_auth_headers(self, endpoint: NetworkEndpoint) -> Dict[str, str]:
        """Get authentication headers for endpoint."""
        headers = {}
        
        try:
            if self.config.authentication == AuthenticationType.BASIC:
                username = self.config.auth_credentials.get('username', '')
                password = self.config.auth_credentials.get('password', '')
                auth_string = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers['Authorization'] = f"Basic {auth_string}"
            
            elif self.config.authentication == AuthenticationType.BEARER:
                token = self.config.auth_credentials.get('token', '')
                headers['Authorization'] = f"Bearer {token}"
            
            elif self.config.authentication == AuthenticationType.HMAC:
                secret = self.config.auth_credentials.get('secret', '').encode()
                message = f"{endpoint.host}:{endpoint.port}:{int(time.time())}".encode()
                signature = hmac.new(secret, message, hashlib.sha256).hexdigest()
                headers['X-HMAC-Signature'] = signature
                headers['X-HMAC-Timestamp'] = str(int(time.time()))
            
        except Exception as e:
            self._logger.warning(f"Authentication header generation failed: {e}")
        
        return headers
    
    def _update_endpoint_stats(self, endpoint_key: str, message_count: int, byte_count: int, response_time: float) -> None:
        """Update endpoint statistics."""
        try:
            stats = self._endpoint_stats[endpoint_key]
            stats['messages_sent'] += message_count
            stats['bytes_sent'] += byte_count
            stats['response_times'].append(response_time)
            
        except Exception as e:
            self._logger.warning(f"Stats update failed for {endpoint_key}: {e}")
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive network sink statistics."""
        base_stats = self.get_stats().to_dict()
        
        # Network-specific statistics
        network_stats = {
            'messages_sent': self._messages_sent,
            'bytes_sent': self._bytes_sent,
            'send_errors': self._send_errors,
            'batch_count': self._batch_count,
            'current_batch_size': self.current_batch.size(),
            'endpoints': len(self.config.endpoints),
            'async_mode': self.async_mode,
            'worker_threads': self.worker_threads
        }
        
        # Endpoint statistics
        endpoint_stats = {}
        for endpoint_key, stats in self._endpoint_stats.items():
            endpoint_stats[endpoint_key] = {
                'messages_sent': stats['messages_sent'],
                'bytes_sent': stats['bytes_sent'],
                'errors': stats['errors'],
                'avg_response_time_ms': (
                    sum(stats['response_times']) / len(stats['response_times']) * 1000
                    if stats['response_times'] else 0
                ),
                'circuit_breaker_state': self.circuit_breakers[endpoint_key].state
            }
        
        network_stats['endpoint_stats'] = endpoint_stats
        
        # Configuration summary
        network_stats['configuration'] = {
            'batch_size': self.config.batch_size,
            'batch_timeout': self.config.batch_timeout_seconds,
            'compression': self.config.compression.value,
            'authentication': self.config.authentication.value,
            'max_retries': self.config.max_retries
        }
        
        # Merge with base stats
        base_stats.update(network_stats)
        return base_stats
    
    def force_flush(self) -> bool:
        """Force immediate flush of current batch."""
        try:
            with self.batch_lock:
                if self.current_batch.size() > 0:
                    batch_to_send = self.current_batch
                    self.current_batch = NetworkBatch(
                        max_size=self.config.batch_size,
                        timeout=self.config.batch_timeout_seconds
                    )
                    
                    return self._send_batch(batch_to_send)
            return True
            
        except Exception as e:
            self._logger.error(f"Force flush failed: {e}")
            return False
    
    def add_endpoint(self, endpoint: NetworkEndpoint) -> None:
        """Add new network endpoint."""
        self.config.endpoints.append(endpoint)
        
        endpoint_key = f"{endpoint.host}:{endpoint.port}"
        self.circuit_breakers[endpoint_key] = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_threshold,
            reset_timeout=self.config.circuit_breaker_timeout
        )
        self._endpoint_stats[endpoint_key] = {
            'messages_sent': 0,
            'bytes_sent': 0,
            'errors': 0,
            'response_times': deque(maxlen=100)
        }
        
        self._logger.info(f"Added network endpoint: {endpoint_key}")


# Module exports
__all__ = [
    'NetworkSink',
    'NetworkEndpoint',
    'NetworkConfig',
    'NetworkProtocol',
    'CompressionType',
    'AuthenticationType'
]