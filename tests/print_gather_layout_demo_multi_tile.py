"""Demo: multi-tile (expert, rank) chunks with padded last tile.

Hand-crafts a routing_topk so that several (expert, rank) chunks span more
than one BLOCK_M-tile AND the last tile of each chunk has nonzero padding.
This stresses the "ring-order with multi-tile per chunk + per-chunk pad
on the trailing tile" semantics.
"""

import torch
import deep_gemm

# ---------------------------------------------------------------------------
# Config — chosen so each (e, r) chunk has a known, hand-picked size that
# exceeds BLOCK_M and is not a multiple of BLOCK_M.
# ---------------------------------------------------------------------------
NUM_RANKS       = 2
TOKENS_PER_RANK = 16        # tokens 0..15 belong to rank 0, 16..31 to rank 1
NUM_EXPERTS     = 2
TOP_K           = 1         # each token picks exactly one expert
BLOCK_M         = 4
LOCAL_RANK      = 0         # ring order from rank 0: [0, 1]

T = NUM_RANKS * TOKENS_PER_RANK


# ---------------------------------------------------------------------------
# Hand-crafted routing so that count[e][r] is exactly:
#
#   count[e=0][r=0] = 10   (tokens 0..9 pick expert 0)
#   count[e=1][r=0] =  6   (tokens 10..15 pick expert 1)
#   count[e=0][r=1] =  7   (tokens 16..22 pick expert 0)
#   count[e=1][r=1] =  9   (tokens 23..31 pick expert 1)
#
# Expected n_slot / n_real / n_pad per chunk (BLOCK_M=4):
#
#   (e=0, r=0): 10 → 3 tiles, last tile = 2 real + 2 pad
#   (e=0, r=1):  7 → 2 tiles, last tile = 3 real + 1 pad     ← multi-tile + pad
#   (e=1, r=0):  6 → 2 tiles, last tile = 2 real + 2 pad     ← multi-tile + pad
#   (e=1, r=1):  9 → 3 tiles, last tile = 1 real + 3 pad     ← multi-tile + pad
# ---------------------------------------------------------------------------
routing_topk = torch.empty((T, TOP_K), dtype=torch.int32, device='cuda')
routing_topk[0:10, 0]  = 0   # rank 0, 10 tokens -> expert 0
routing_topk[10:16, 0] = 1   # rank 0,  6 tokens -> expert 1
routing_topk[16:23, 0] = 0   # rank 1,  7 tokens -> expert 0
routing_topk[23:32, 0] = 1   # rank 1,  9 tokens -> expert 1


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

gi = gather_index[:m_logical].cpu().tolist()
gl = grouped_layout[:m_logical].cpu().tolist()
tr = tile_rank[: m_logical // BLOCK_M].cpu().tolist()


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------
def hr(): print('=' * 90)

hr()
print('  Inputs')
hr()
print(f'  num_ranks       = {NUM_RANKS}')
print(f'  tokens_per_rank = {TOKENS_PER_RANK}    (T = {T})')
print(f'  num_experts     = {NUM_EXPERTS}')
print(f'  top_k           = {TOP_K}')
print(f'  block_m         = {BLOCK_M}')
print(f'  local_rank      = {LOCAL_RANK}    (ring order: '
      f'{[(LOCAL_RANK + s) % NUM_RANKS for s in range(NUM_RANKS)]})')
print()


# Histogram cross-check
counts = [[0] * NUM_RANKS for _ in range(NUM_EXPERTS)]
rt_cpu = routing_topk.cpu().tolist()
for t in range(T):
    r = t // TOKENS_PER_RANK
    for j in range(TOP_K):
        counts[rt_cpu[t][j]][r] += 1

hr()
print('  Hand-crafted counts[expert][rank] (Phase 1 will produce these):')
hr()
print('     expert \\ rank  ' + '  '.join(f'r={r}' for r in range(NUM_RANKS)))
for e in range(NUM_EXPERTS):
    print(f'        e={e}          '
          + '   '.join(f'{counts[e][r]:2d}' for r in range(NUM_RANKS)))
print()

hr()
print('  Layout-A chunk plan')
hr()
print('   m_offset | tile_idx | (e, s, rank) | n_real | n_pad | n_slot | n_tiles | layout')
print('  ' + '-' * 88)
cum = 0
tile_idx = 0
for e in range(NUM_EXPERTS):
    for s in range(NUM_RANKS):
        r = (LOCAL_RANK + s) % NUM_RANKS
        n_real = counts[e][r]
        n_slot = ((n_real + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
        n_pad  = n_slot - n_real
        n_tiles = n_slot // BLOCK_M
        last_tile_real = n_real - (n_tiles - 1) * BLOCK_M if n_tiles >= 1 else 0
        last_tile_pad  = BLOCK_M - last_tile_real if n_tiles >= 1 else 0
        layout_str = (f'{n_tiles - 1} full tile + last tile = ({last_tile_real} real + {last_tile_pad} pad)'
                      if n_tiles > 1 else
                      f'1 tile = ({last_tile_real} real + {last_tile_pad} pad)')
        marker = '   ← multi-tile + pad' if n_tiles > 1 and n_pad > 0 else ''
        print(f'    {cum:5d}   |  {tile_idx:5d}   | (e={e}, s={s}, r={r}) |  {n_real:3d}   |  '
              f'{n_pad:3d}  |  {n_slot:3d}   |   {n_tiles}    | {layout_str}{marker}')
        cum += n_slot
        tile_idx += n_tiles
print(f'  m_logical = {cum}, num_m_tiles = {tile_idx}')
print()


hr()
print('  Generator outputs')
hr()
print(f'  m_logical = {m_logical}')
print()
print(f'  tile_rank[:{m_logical // BLOCK_M}] = {tr}')
print()


def _fmt_block(arr, block_m):
    blocks = [arr[i:i + block_m] for i in range(0, len(arr), block_m)]
    return '[' + ' | '.join(' '.join(f'{x:3d}' for x in b) for b in blocks) + ']'


print('  gather_index (one bracket = 1 m-tile of BLOCK_M=' + str(BLOCK_M) + ' rows; -1 = pad):')
print(f'    {_fmt_block(gi, BLOCK_M)}')
print()
print('  grouped_layout (per-row expert; pad rows hold chunk\'s expert id):')
print(f'    {_fmt_block(gl, BLOCK_M)}')
print()


# ---------------------------------------------------------------------------
# Per-tile breakdown (the most useful view for showing multi-tile chunks)
# ---------------------------------------------------------------------------
hr()
print('  Per-tile breakdown')
hr()
print('   tile_idx | tile_rank | expert | gather_index in this tile        | tile composition')
print('  ' + '-' * 85)

# Walk through chunks to annotate which tile is "first / mid / last in chunk".
chunk_tiles = []      # (e, r, tile_in_chunk, n_tiles_in_chunk)
for e in range(NUM_EXPERTS):
    for s in range(NUM_RANKS):
        r = (LOCAL_RANK + s) % NUM_RANKS
        n_real = counts[e][r]
        n_slot = ((n_real + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
        n_tiles = n_slot // BLOCK_M
        for k in range(n_tiles):
            chunk_tiles.append((e, r, k, n_tiles))

for t_idx in range(m_logical // BLOCK_M):
    base = t_idx * BLOCK_M
    rows = gi[base: base + BLOCK_M]
    expert = gl[base]
    e, r, k_in_chunk, n_in_chunk = chunk_tiles[t_idx]
    real_count = sum(1 for x in rows if x >= 0)
    pad_count = BLOCK_M - real_count
    if n_in_chunk == 1:
        role = 'only tile in (e=%d, r=%d) chunk' % (e, r)
    elif k_in_chunk == 0:
        role = 'first of %d tiles in (e=%d, r=%d)' % (n_in_chunk, e, r)
    elif k_in_chunk == n_in_chunk - 1:
        role = '** LAST of %d tiles in (e=%d, r=%d) **' % (n_in_chunk, e, r)
    else:
        role = 'tile %d/%d in (e=%d, r=%d)' % (k_in_chunk + 1, n_in_chunk, e, r)
    composition = f'{real_count} real + {pad_count} pad — {role}'
    print(f'     {t_idx:4d}   |    r={tr[t_idx]}    |   e={expert}  | {str(rows):32s} | {composition}')
print()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
hr()
print('  Verification: reconstruct routing_topk from gather_index')
hr()
hits, errors = 0, 0
for pos in range(m_logical):
    src_t = gi[pos]
    if src_t < 0:
        continue
    pred_rank = src_t // TOKENS_PER_RANK
    pred_expert = gl[pos]
    tile_r = tr[pos // BLOCK_M]
    if pred_rank == tile_r and pred_expert in rt_cpu[src_t]:
        hits += 1
    else:
        errors += 1
        print(f'    [BAD] pos={pos}: src_t={src_t}, expert={pred_expert}, '
              f'tile_rank={tile_r}, predicted_rank={pred_rank}')
total_pairs = T * TOP_K
n_pad = gi.count(-1)
print(f'    {hits} / {total_pairs} (token, expert) pairs accounted for; {errors} mismatches')
print(f'    {n_pad} pad rows; {m_logical - n_pad} real rows == T * top_k = {total_pairs}? '
      f'{m_logical - n_pad == total_pairs}')
hr()
