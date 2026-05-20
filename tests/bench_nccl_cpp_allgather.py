"""Microbench the direct C++ ncclAllGather wrapper without torch.distributed.

This is intended to compare the Python-imported C++ NCCL op against
nccl-tests. It avoids creating a PyTorch ProcessGroupNCCL communicator.
"""

import argparse
import glob
import os
import statistics
import sys
import time
import uuid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.multiprocessing as mp

import deep_gemm


def _file_barrier(sync_dir: str, name: str, rank: int, num_ranks: int) -> None:
    os.makedirs(sync_dir, exist_ok=True)
    path = os.path.join(sync_dir, f'{name}.{rank}')
    with open(path, 'w') as f:
        f.write('1')

    pattern = os.path.join(sync_dir, f'{name}.*')
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if len(glob.glob(pattern)) >= num_ranks:
            return
        time.sleep(0.001)
    raise TimeoutError(f'timed out waiting at barrier {name}')


def _load_or_create_unique_id(local_rank: int, unique_id_path: str) -> bytes:
    if local_rank == 0:
        unique_id = deep_gemm.nccl_get_unique_id()
        tmp_path = f'{unique_id_path}.tmp'
        with open(tmp_path, 'wb') as f:
            f.write(unique_id)
        os.replace(tmp_path, unique_id_path)
        return unique_id

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if os.path.exists(unique_id_path):
            with open(unique_id_path, 'rb') as f:
                return f.read()
        time.sleep(0.001)
    raise TimeoutError(f'timed out waiting for {unique_id_path}')


def _set_default_nccl_ctas() -> None:
    os.environ.setdefault('NCCL_MIN_CTAS', '64')
    os.environ.setdefault('NCCL_MAX_CTAS', '64')


def _run_rank(device_idx: int, comm_rank: int, num_ranks: int, unique_id_path: str,
              args, barrier_fn) -> tuple[float, float]:
    torch.cuda.set_device(device_idx)
    torch.set_default_device(f'cuda:{device_idx}')

    unique_id = _load_or_create_unique_id(comm_rank, unique_id_path)
    comm = deep_gemm.nccl_comm_init_rank(unique_id, comm_rank, num_ranks, device_idx)
    send = torch.empty((args.bytes_per_rank,), dtype=torch.uint8, device='cuda')
    send.fill_(comm_rank)
    recv = torch.empty((args.bytes_per_rank * num_ranks,), dtype=torch.uint8, device='cuda')
    stream = torch.cuda.Stream()

    def enqueue_allgather() -> None:
        with torch.cuda.stream(stream):
            deep_gemm.nccl_allgather_bytes(send, recv, comm)

    if args.cpp_internal_bench:
        barrier_fn()
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            event_ms = deep_gemm.nccl_allgather_bytes_bench(send, recv, comm, args.warmups, args.iters)
        torch.cuda.synchronize()
        barrier_fn()
        cpu_ms = event_ms
        if args.check:
            ok = True
            for rank in range(num_ranks):
                chunk = recv[rank * args.bytes_per_rank:(rank + 1) * args.bytes_per_rank]
                ok = ok and bool(torch.all(chunk == rank).item())
            if not ok:
                raise RuntimeError(f'allgather check failed on rank {comm_rank}')
        return event_ms, cpu_ms

    if args.no_iter_barrier:
        barrier_fn()

    for _ in range(args.warmups):
        if not args.no_iter_barrier:
            barrier_fn()
        enqueue_allgather()
        torch.cuda.synchronize()
        if not args.no_iter_barrier:
            barrier_fn()

    if args.no_iter_barrier:
        barrier_fn()

    event_times = []
    cpu_times = []
    for iter_idx in range(args.iters):
        if not args.no_iter_barrier:
            barrier_fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        t0 = time.perf_counter()
        enqueue_allgather()
        end.record(stream)
        end.synchronize()
        cpu_times.append((time.perf_counter() - t0) * 1e3)
        event_times.append(start.elapsed_time(end))
        if not args.no_iter_barrier:
            barrier_fn()

    if args.check:
        torch.cuda.synchronize()
        ok = True
        for rank in range(num_ranks):
            chunk = recv[rank * args.bytes_per_rank:(rank + 1) * args.bytes_per_rank]
            ok = ok and bool(torch.all(chunk == rank).item())
        if not ok:
            raise RuntimeError(f'allgather check failed on rank {comm_rank}')

    return statistics.median(event_times), statistics.median(cpu_times)


def _worker(local_rank: int, num_ranks: int, unique_id_path: str, args, barrier, queue) -> None:
    barrier_fn = barrier.wait
    event_ms, cpu_ms = _run_rank(local_rank, local_rank, num_ranks, unique_id_path, args, barrier_fn)
    queue.put((local_rank, event_ms, cpu_ms))


def _print_results(results: list[tuple[int, float, float]], num_ranks: int, bytes_per_rank: int,
                   show_cpu_wait: bool = True) -> None:
    results.sort()
    event_values = [x[1] for x in results]
    cpu_values = [x[2] for x in results]
    event_ms = max(event_values)
    cpu_ms = max(cpu_values)
    remote_bytes = (num_ranks - 1) * bytes_per_rank
    output_bytes = num_ranks * bytes_per_rank

    print('Direct C++ NCCL allgather microbench:', flush=True)
    print(f'  ranks={num_ranks}, bytes/rank={bytes_per_rank}, output/rank={output_bytes}', flush=True)
    print(f'  nccl_ctas={os.environ["NCCL_MIN_CTAS"]}/{os.environ["NCCL_MAX_CTAS"]}', flush=True)
    print(f'  event max       : {event_ms * 1e3:8.2f} us', flush=True)
    print(f'  event avg       : {statistics.mean(event_values) * 1e3:8.2f} us', flush=True)
    print(f'  event min       : {min(event_values) * 1e3:8.2f} us', flush=True)
    if show_cpu_wait:
        print(f'  cpu wait max    : {cpu_ms * 1e3:8.2f} us', flush=True)
        print(f'  cpu wait avg    : {statistics.mean(cpu_values) * 1e3:8.2f} us', flush=True)
        print(f'  cpu wait min    : {min(cpu_values) * 1e3:8.2f} us', flush=True)
    else:
        print('  cpu wait        :      n/a (timed inside C++ extension)', flush=True)
    print(f'  remote BW/rank  : {remote_bytes / (event_ms * 1e-3) / 1e9:8.2f} GB/s', flush=True)
    print(f'  output BW/rank  : {output_bytes / (event_ms * 1e-3) / 1e9:8.2f} GB/s', flush=True)


def _external_launch_main(args) -> None:
    num_ranks = int(os.environ.get('SLURM_NTASKS') or os.environ.get('WORLD_SIZE') or '1')
    comm_rank = int(os.environ.get('SLURM_PROCID') or os.environ.get('RANK') or '0')
    local_rank = int(os.environ.get('SLURM_LOCALID') or os.environ.get('LOCAL_RANK') or comm_rank)
    if args.sync_dir is None:
        job_id = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
        args.sync_dir = os.path.join('/tmp', f'deep_gemm_nccl_ext_{job_id}')
    os.makedirs(args.sync_dir, exist_ok=True)

    device_idx = 0 if torch.cuda.device_count() == 1 else local_rank
    unique_id_path = os.path.join(args.sync_dir, 'unique_id')

    barrier_count = {'value': 0}
    def barrier_fn() -> None:
        name = f'barrier_{barrier_count["value"]:05d}'
        barrier_count['value'] += 1
        _file_barrier(args.sync_dir, name, comm_rank, num_ranks)

    event_ms, cpu_ms = _run_rank(device_idx, comm_rank, num_ranks, unique_id_path, args, barrier_fn)
    result_path = os.path.join(args.sync_dir, f'result.{comm_rank}')
    with open(result_path, 'w') as f:
        f.write(f'{event_ms} {cpu_ms}\n')

    if comm_rank == 0:
        deadline = time.monotonic() + 120.0
        pattern = os.path.join(args.sync_dir, 'result.*')
        while time.monotonic() < deadline and len(glob.glob(pattern)) < num_ranks:
            time.sleep(0.001)
        result_files = glob.glob(pattern)
        if len(result_files) < num_ranks:
            raise TimeoutError('timed out waiting for rank result files')
        results = []
        for path in result_files:
            rank = int(os.path.basename(path).split('.')[-1])
            with open(path) as f:
                event_s, cpu_s = f.read().split()
            results.append((rank, float(event_s), float(cpu_s)))
        _print_results(results, num_ranks, args.bytes_per_rank,
                       show_cpu_wait=not args.cpp_internal_bench)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-local-ranks', type=int, default=None)
    parser.add_argument('--bytes-per-rank', type=int, default=7351 * 2048)
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--external-launch', action='store_true',
                        help='Run as one externally launched rank, e.g. srun --ntasks=8 --gpus-per-task=1')
    parser.add_argument('--no-iter-barrier', action='store_true',
                        help='Use one barrier around warmup/measurement instead of synchronizing every iteration')
    parser.add_argument('--cpp-internal-bench', action='store_true',
                        help='Run warmup/iters/CUDA events inside the C++ extension to avoid Python per-iter launch gaps')
    parser.add_argument('--sync-dir', type=str, default=None,
                        help='Directory for file-based unique-id/result exchange in --external-launch mode')
    args = parser.parse_args()

    _set_default_nccl_ctas()

    if args.external_launch:
        _external_launch_main(args)
        return

    num_ranks = args.num_local_ranks or torch.cuda.device_count()
    if num_ranks <= 0:
        raise RuntimeError('no CUDA devices found')

    unique_id_path = os.path.join('/tmp', f'deep_gemm_nccl_{uuid.uuid4().hex}.uid')
    ctx = mp.get_context('spawn')
    barrier = ctx.Barrier(num_ranks)
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(rank, num_ranks, unique_id_path, args, barrier, queue))
        for rank in range(num_ranks)
    ]

    try:
        for proc in procs:
            proc.start()
        results = [queue.get() for _ in procs]
        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f'worker process {proc.pid} exited with code {proc.exitcode}')
    finally:
        if os.path.exists(unique_id_path):
            os.remove(unique_id_path)

    _print_results(results, num_ranks, args.bytes_per_rank,
                   show_cpu_wait=not args.cpp_internal_bench)


if __name__ == '__main__':
    main()
