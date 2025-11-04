import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
env_path = os.path.join(BASE_DIR, ".env")

load_dotenv(env_path)

# Server socket
bind = f"0.0.0.0:{os.getenv('FLASK_PORT', 5000)}"
backlog = 2048

# Worker processes
workers = int(os.getenv("GUNICORN_WORKERS", 2))
worker_class = "gthread"
threads = int(os.getenv("GUNICORN_THREADS", 2))
worker_connections = 1000
timeout = 180
keepalive = 2

preload_app = False


# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"  # Log to stderr
loglevel = os.getenv("LOG_LEVEL", "info")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "nilm_flask_app"

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None


# Worker timeout
graceful_timeout = 30


def worker_int(worker):
    """Handle worker interruption (for graceful MQTT cleanup)"""
    worker.log.info("Worker received INT or QUIT signal")
