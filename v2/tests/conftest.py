import os

os.environ.setdefault("KMP_BLOCKTIME", "0")

from threadpoolctl import threadpool_limits  # noqa: E402

# one OpenMP thread keeps model fits fast and deterministic on shared machines
_LIMIT = threadpool_limits(limits=1)
