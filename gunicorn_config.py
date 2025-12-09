# Gunicorn configuration file - Optimized for low memory environments
import multiprocessing

# Server socket
bind = "0.0.0.0:10000"
backlog = 512  # Reduced from 2048

# Worker processes - CRITICAL for memory management
workers = 1  # Single worker for free tier (512MB RAM)
worker_class = 'sync'
worker_connections = 100  # Reduced from 1000
timeout = 120  # 2 minutes - reduced from 5 to prevent hanging
keepalive = 2

# Memory management - AGGRESSIVE settings to prevent OOM
max_requests = 50  # Restart worker after 50 requests (was 100)
max_requests_jitter = 10
worker_tmp_dir = '/dev/shm'  # Use RAM disk for worker heartbeat

# Preload app to save memory
preload_app = False  # Set to False to avoid memory spikes on restart

# Graceful timeout
graceful_timeout = 30

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Process naming
proc_name = 'dukhtar_app'
