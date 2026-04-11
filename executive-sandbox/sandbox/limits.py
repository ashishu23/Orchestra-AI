"""Resource limit constants for the Python execution sandbox."""

MAX_CPU_SECONDS: int = 10       # Hard CPU time limit per execution
MAX_MEMORY_MB: int = 128        # Virtual memory cap per execution
TIMEOUT_SECONDS: int = 15       # Wall-clock timeout (includes I/O wait)

# Derived byte values used with resource.setrlimit
MAX_MEMORY_BYTES: int = MAX_MEMORY_MB * 1024 * 1024
