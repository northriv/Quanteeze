"""Shared pytest setup."""

from quanteeze import serve

# The server installs a `/proc/meminfo` free-RAM reader into `mlx_vlm.apc` at startup
# (mlx_vlm only knows psutil and macOS `vm_stat`). Mirror that here so APC tests on
# Linux without psutil see the same memory headroom as the running server.
serve._apc_reads_free_memory()
