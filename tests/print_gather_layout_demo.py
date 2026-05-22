"""Minimal hand-traceable demo of `build_gather_layout_for_rank_overlap`.

Prints `routing_topk`, `tile_rank`, `gather_index`, `grouped_layout`, and
`m_logical` for a small controlled input so the layout-A semantics can be
verified manually against `docs/sm90_fp8_gemm_1d2d_gather_index_rank_overlap.md`.
"""

import os
import torch

import deep_gemm

# ---------------------------------------------------------------------------
# Tiny config so everything fits on one screen
# ---------------------------------------------------------------------------
NUM_RANKS       = 4
TOKENS_PER_RANK = 4         # T = num_ranks * tokens_per_rank = 16
NUM_EXPERTS     = 4
TOP_K           = 2
BLOCK_M         = 4
LOCAL_RANK      = 1         # ring order from rank 1: [1, 2, 3, 0]

T = NUM_RANKS * TOKENS_PER_RANK


# ---------------------------------------------------------------------------
# Deterministic routing_topk so the run is reproducible.
#
# We hand-craft it so that token t selects experts
#   ((t * 3 + 0) mod NUM_EXPERTS, (t * 3 + 1) mod NUM_EXPERTS),
# guaranteed-distinct since num_experts > top_k.
# ---------------------------------------------------------------------------
routing_topk = torch.empty((T, TOP_K), dtype=torch.int32, device='cuda')
for t in range(T):
    for j in range(TOP_K):
        routing_topk[t, j] = (t * 3 + j) % NUM_EXPERTS


# ---------------------------------------------------------------------------
# Run the generator
# ---------------------------------------------------------------------------
gather_index, tile_rank, grouped_layout, m_logical_t, psum_layout, row_to_topk = \
    deep_gemm.build_gather_layout_for_rank_overlap(
        routing_topk,
        local_rank=LOCAL_RANK,
        num_ranks=NUM_RANKS,
        tokens_per_rank=TOKENS_PER_RANK,
        num_experts=NUM_EXPERTS,
        block_m=BLOCK_M,
    )
torch.cuda.synchronize()
m_logical = int(m_logical_t.item())

# Truncate to the actual valid prefix (the buffers are over-allocated to M_max).
gi = gather_index[:m_logical].cpu().tolist()
gl = grouped_layout[:m_logical].cpu().tolist()
tr = tile_rank[: m_logical // BLOCK_M].cpu().tolist()


# ---------------------------------------------------------------------------
# Print everything
# ---------------------------------------------------------------------------
def hr(): print('=' * 78)

hr()
print('  Inputs')
hr()
print(f'  num_ranks       = {NUM_RANKS}')
print(f'  tokens_per_rank = {TOKENS_PER_RANK}        (T = num_ranks * tpr = {T})')
print(f'  num_experts     = {NUM_EXPERTS}')
print(f'  top_k           = {TOP_K}')
print(f'  block_m         = {BLOCK_M}')
print(f'  local_rank      = {LOCAL_RANK}        (ring order: '
      f'{[(LOCAL_RANK + s) % NUM_RANKS for s in range(NUM_RANKS)]})')
print()
print('  routing_topk (each row = top-k expert choices for token t):')
print('     token  rank | top-k experts')
print('     ' + '-' * 35)
rt_cpu = routing_topk.cpu().tolist()
for t in range(T):
    r = t // TOKENS_PER_RANK
    print(f'      t={t:2d}  r={r}  | {rt_cpu[t]}')
print()


# ---------------------------------------------------------------------------
# Histogram counts[e][r] — derived from routing_topk for cross-check.
# ---------------------------------------------------------------------------
counts = [[0] * NUM_RANKS for _ in range(NUM_EXPERTS)]
for t in range(T):
    r = t // TOKENS_PER_RANK
    for j in range(TOP_K):
        counts[rt_cpu[t][j]][r] += 1

hr()
print('  Phase-1 derived: counts[expert][rank]')
hr()
print('     expert \\ rank ' + ' '.join(f'r={r}' for r in range(NUM_RANKS)))
for e in range(NUM_EXPERTS):
    print(f'        e={e}        ' + '   '.join(f'{counts[e][r]:1d}' for r in range(NUM_RANKS)))
print()
print('  Layout-A chunk plan (for local_rank = '
      f'{LOCAL_RANK} → ring step s → rank (LOCAL_RANK+s) % {NUM_RANKS}):')
print('  ' + '-' * 70)
print('   m_offset | tile_idx | (e, s, rank) | n_real | n_pad | n_slot | gather_index slice')
print('  ' + '-' * 70)
cum = 0
tile_idx = 0
for e in range(NUM_EXPERTS):
    for s in range(NUM_RANKS):
        r = (LOCAL_RANK + s) % NUM_RANKS
        n_real = counts[e][r]
        n_slot = ((n_real + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
        n_pad = n_slot - n_real
        n_tiles = n_slot // BLOCK_M
        slice_repr = f'[{cum}, {cum + n_real}) real + [{cum + n_real}, {cum + n_slot}) pad'
        print(f'    {cum:6d}  |  {tile_idx:5d}   | (e={e}, s={s}, r={r}) |   {n_real}    |   '
              f'{n_pad}   |   {n_slot}    | {slice_repr}')
        cum += n_slot
        tile_idx += n_tiles
print(f'  m_logical = {cum}, num_m_tiles = {tile_idx}')
print()


# ---------------------------------------------------------------------------
# Generator outputs
# ---------------------------------------------------------------------------
hr()
print('  Generator outputs')
hr()
print(f'  m_logical             = {m_logical}')
print()
print(f'  tile_rank[:{m_logical // BLOCK_M}]    = {tr}')
print(f'    expected (layout A)  = '
      f'{sum([[((LOCAL_RANK + s) % NUM_RANKS)] * (((counts[e][(LOCAL_RANK + s) % NUM_RANKS] + BLOCK_M - 1) // BLOCK_M)) for e in range(NUM_EXPERTS) for s in range(NUM_RANKS)], [])}')
print()


def _fmt_block(arr, block_m):
    blocks = [arr[i:i + block_m] for i in range(0, len(arr), block_m)]
    return '[' + ' | '.join(' '.join(f'{x:3d}' for x in b) for b in blocks) + ']'


print('  gather_index  (one bracket per BLOCK_M=' + str(BLOCK_M) + ' tile, "-1" = pad row):')
print(f'    {_fmt_block(gi, BLOCK_M)}')
print()
print('  grouped_layout (per-row expert id; pad rows still hold chunk\'s expert id):')
print(f'    {_fmt_block(gl, BLOCK_M)}')
print()


# ---------------------------------------------------------------------------
# Per-tile annotation (the most useful view)
# ---------------------------------------------------------------------------
hr()
print('  Per-tile breakdown (tile = block of BLOCK_M rows)')
hr()
print('    tile_idx | tile_rank | rows in this tile (gather_index)             | grouped_layout (expert id)')
print('    ' + '-' * 96)
for t_idx in range(m_logical // BLOCK_M):
    base = t_idx * BLOCK_M
    rows_gi = gi[base: base + BLOCK_M]
    rows_gl = gl[base: base + BLOCK_M]
    print(f'      {t_idx:4d}  |    r={tr[t_idx]}    | {rows_gi}'
          f' | {rows_gl}')
print()


# ---------------------------------------------------------------------------
# Cross-check: every (token, expert) pair from routing_topk should appear
# exactly once in gather_index, in a position whose grouped_layout matches the
# expert id and whose tile_rank matches the token's rank.
# ---------------------------------------------------------------------------
hr()
print('  Verification: scatter-back reconstruct routing_topk')
hr()
hits = 0
errors = 0
for pos in range(m_logical):
    src_t = gi[pos]
    if src_t < 0:
        continue
    pred_rank = src_t // TOKENS_PER_RANK
    pred_expert = gl[pos]
    tile_idx = pos // BLOCK_M
    tile_r = tr[tile_idx]
    ok_rank = (pred_rank == tile_r)
    ok_expert = pred_expert in rt_cpu[src_t]
    if ok_rank and ok_expert:
        hits += 1
    else:
        errors += 1
        print(f'    [BAD] pos={pos}, src_t={src_t}, expert={pred_expert}, '
              f'tile_rank={tile_r}, predicted_rank={pred_rank}, '
              f'token_topk={rt_cpu[src_t]}')
total_pairs = T * TOP_K
print(f'    {hits} / {total_pairs} (token, expert) pairs accounted for; '
      f'{errors} mismatches')
print(f'    {gi.count(-1)} pad rows; '
      f'{m_logical - gi.count(-1)} real rows == T * top_k = {total_pairs}? '
      f'{m_logical - gi.count(-1) == total_pairs}')
hr()
