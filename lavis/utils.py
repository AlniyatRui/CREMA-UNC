"""
Utility functions for debugging in distributed multi-GPU environments.
"""

import torch
from torch import distributed as dist
    
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import os
import pdb
import sys


__all__ = ["set_trace"]


_stdin = [None]
_stdin_lock = multiprocessing.Lock()
try:
    _stdin_fd = sys.stdin.fileno()
except Exception:
    _stdin_fd = None


def breakpoint_rank0():
    """
    Set a breakpoint only on the process with rank 0,
    and synchronize all processes afterward using a barrier.
    
    This function can be imported and used anywhere in your code.
    
    Usage:
        from utils import breakpoint_rank0
        
        # Your distributed training code...
        breakpoint_rank0()  # Only rank 0 will pause for debugging
    """
    if dist.get_rank() == 0:
        import pdb
        pdb.set_trace()
    dist.barrier()


def breakpoint_at_rank(rank=0):
    """
    Set a breakpoint only on the process with the specified rank,
    and synchronize all processes afterward using a barrier.
    
    Args:
        rank (int): The rank of the process where the breakpoint should be set.
                    Default is 0.
    
    Usage:
        from utils import breakpoint_at_rank
        
        # Your distributed training code...
        breakpoint_at_rank(1)  # Only rank 1 will pause for debugging
    """
    if dist.get_rank() == rank:
        import pdb
        pdb.set_trace()
    dist.barrier()


def is_distributed():
    """
    Check if the current process is running in a distributed environment.
    
    Returns:
        bool: True if running in a distributed environment, False otherwise.
    """
    try:
        return dist.is_initialized()
    except:
        return False


def safe_breakpoint_rank0():
    """
    Set a breakpoint only on the process with rank 0 if in a distributed environment.
    If not in a distributed environment, set a regular breakpoint.
    No barrier is used if not in a distributed environment.
    
    Usage:
        from utils import safe_breakpoint_rank0
        
        # Your code that might run in both distributed and non-distributed modes
        safe_breakpoint_rank0()
    """
    if is_distributed():
        if dist.get_rank() == 0:
            import pdb
            pdb.set_trace()
        dist.barrier()
    else:
        import pdb
        pdb.set_trace()


class MultiprocessingPdb(pdb.Pdb):
    """A Pdb wrapper that works in a multiprocessing environment.

    Usage: `from fairseq import pdb; pdb.set_trace()`
    """

    def __init__(self):
        pdb.Pdb.__init__(self, nosigint=True)

    def _cmdloop(self):
        stdin_bak = sys.stdin
        with _stdin_lock:
            try:
                if _stdin_fd is not None:
                    if not _stdin[0]:
                        _stdin[0] = os.fdopen(_stdin_fd)
                    sys.stdin = _stdin[0]
                self.cmdloop()
            finally:
                sys.stdin = stdin_bak


def set_trace():
    pdb = MultiprocessingPdb()
    pdb.set_trace(sys._getframe().f_back)