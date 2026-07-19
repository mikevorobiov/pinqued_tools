import numba as _numba

# The default 'workqueue' threading layer is not safe when parallel=True
# jitted functions are invoked concurrently from multiple Python threads
# (e.g. a ThreadPoolExecutor bootstrap loop) -- it can corrupt internal
# buffer sizing and crash with a spurious MemoryError/SystemError.
# 'threadsafe' picks tbb (or omp) instead, which supports that use case.
_numba.config.THREADING_LAYER = "threadsafe"
