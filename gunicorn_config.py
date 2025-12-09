# Gunicorn configuration file
import multiprocessing

# Server socket
bind = "0.0.0.0:10000"
backlog = 2048

# Worker processes
workers = 1  # Use only 1 worker to save memory on free tier
worker_class = 'sync'
worker_connections = 1000
timeout = 300  # Increase timeout to 5 minutes for AI generation
keepalive = 2

# Memory management
max_requests = 100  # Restart worker after 100 requests to prevent memory leaks
max_requests_jitter = 10

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
