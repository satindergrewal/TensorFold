"""GLM's universal EXL3 route uses buffer-owned scratch and the existing bf16 expert epilogue."""

import copy

from tensorfold.cuda.exl3 import experts as generic

# Rows a call groups at once: the grouping kernel holds a window's picks in shared memory (rows x slots x 4 bytes),
# 2,048 rows of GLM's 9 slots = 72 KiB, under every supported GPU's per-block limit. A larger window (a 4,096-row
# prompt chunk, TF_GLM_PREFILL_ROWS) runs in pieces of this many rows: rows never depend on the window, so the bits
# are the one call's.
PIECE_ROWS = 2048


def routed(x, pick, ex, scratch, rows, limit):
    """Write the live routed pairs into scratch.y; the caller supplies each row's shared-expert slot."""

    if rows <= PIECE_ROWS:
        return generic.routed(x, pick, None, ex, scratch, None, rows, limit, act_mode=generic.ACT_BF16)
    slots = scratch.slots
    for lo in range(0, rows, PIECE_ROWS):
        hi = min(rows, lo + PIECE_ROWS)
        piece = copy.copy(scratch)               # the same buffers; this piece's rows of y
        piece.y = scratch.y[lo * slots:hi * slots]
        generic.routed(x[lo:hi], pick[lo:hi], None, ex, piece, None, hi - lo, limit, act_mode=generic.ACT_BF16)
    return scratch.y[:rows * slots]
