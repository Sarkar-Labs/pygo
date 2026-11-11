"""PyGo DataLoader — high-throughput, multi-worker, prefetching data pipeline."""
from __future__ import annotations

import os
import math
import queue
import random
import threading
import multiprocessing as mp
from dataclasses import dataclass
from typing import Iterator, Optional, Callable, Any, List, Sequence, Union

import numpy as np


# ─── Dataset ──────────────────────────────────────────────────────────────────

class Dataset:
    """Map-style dataset: implement __len__ and __getitem__."""
    def __len__(self) -> int:
        raise NotImplementedError
    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError


class IterableDataset:
    """Stream-style dataset for data too large to index (web-scale corpora)."""
    def __iter__(self) -> Iterator:
        raise NotImplementedError


class ConcatDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.cum_sizes = np.cumsum([len(d) for d in datasets])

    def __len__(self):
        return int(self.cum_sizes[-1]) if len(self.cum_sizes) else 0

    def __getitem__(self, idx):
        ds_idx = int(np.searchsorted(self.cum_sizes, idx, side="right"))
        prev   = self.cum_sizes[ds_idx - 1] if ds_idx > 0 else 0
        return self.datasets[ds_idx][idx - prev]


class SubsetDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.dataset[self.indices[idx]]


def random_split(dataset: Dataset, lengths: List[int], seed: int = 42) -> List[SubsetDataset]:
    if sum(lengths) != len(dataset):
        raise ValueError(f"Sum of lengths {sum(lengths)} != dataset size {len(dataset)}")
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    out, offset = [], 0
    for length in lengths:
        out.append(SubsetDataset(dataset, indices[offset:offset+length]))
        offset += length
    return out


# ─── Samplers ─────────────────────────────────────────────────────────────────

class Sampler:
    def __iter__(self) -> Iterator[int]: raise NotImplementedError
    def __len__(self) -> int: raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, data_source): self.n = len(data_source)
    def __iter__(self): return iter(range(self.n))
    def __len__(self): return self.n


class RandomSampler(Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None, seed=None):
        self.n = len(data_source)
        self.replacement = replacement
        self.num_samples  = num_samples or self.n
        self.rng = random.Random(seed)

    def __iter__(self):
        if self.replacement:
            return iter(self.rng.choices(range(self.n), k=self.num_samples))
        indices = list(range(self.n))
        self.rng.shuffle(indices)
        return iter(indices[: self.num_samples])

    def __len__(self): return self.num_samples


class WeightedRandomSampler(Sampler):
    def __init__(self, weights: Sequence[float], num_samples: int,
                 replacement: bool = True, seed: Optional[int] = None):
        self.weights     = np.array(weights, dtype=np.float64)
        self.weights    /= self.weights.sum()
        self.num_samples = num_samples
        self.replacement = replacement
        self.rng         = np.random.RandomState(seed)

    def __iter__(self):
        idx = self.rng.choice(len(self.weights), size=self.num_samples,
                              replace=self.replacement, p=self.weights)
        return iter(idx.tolist())

    def __len__(self): return self.num_samples


class DistributedSampler(Sampler):
    """Splits dataset across `num_replicas` processes, used with DDP."""
    def __init__(self, dataset, num_replicas: int, rank: int,
                 shuffle: bool = True, seed: int = 0, drop_last: bool = False):
        self.dataset      = dataset
        self.num_replicas = num_replicas
        self.rank         = rank
        self.shuffle      = shuffle
        self.seed         = seed
        self.epoch        = 0
        self.drop_last    = drop_last

        n = len(dataset)
        if drop_last and n % num_replicas != 0:
            self.num_samples = math.ceil((n - num_replicas) / num_replicas)
        else:
            self.num_samples = math.ceil(n / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            g = random.Random(self.seed + self.epoch)
            indices = list(range(len(self.dataset)))
            g.shuffle(indices)
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            pad = self.total_size - len(indices)
            indices += indices[:pad]
        else:
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self): return self.num_samples


class BatchSampler(Sampler):
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False):
        self.sampler    = sampler
        self.batch_size = batch_size
        self.drop_last  = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        n = len(self.sampler)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)


# ─── Collate ──────────────────────────────────────────────────────────────────

def default_collate(batch: List[Any]) -> Any:
    """Stack a list of samples into a batched tensor / structure."""
    elem = batch[0]

    if isinstance(elem, np.ndarray):
        return np.stack(batch, axis=0)

    if isinstance(elem, (int, float)):
        return np.array(batch)

    if isinstance(elem, dict):
        return {key: default_collate([d[key] for d in batch]) for key in elem}

    if isinstance(elem, (list, tuple)):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    return batch  # fallback: return as-is


def pad_collate(pad_value: int = 0, pad_keys: Optional[List[str]] = None):
    """Collate fn for variable-length sequences — pads to max length in batch."""
    def _collate(batch: List[dict]) -> dict:
        keys = pad_keys or [k for k, v in batch[0].items()
                            if isinstance(v, (list, np.ndarray))]
        out = {}
        for key in batch[0]:
            if key in keys:
                seqs    = [np.asarray(b[key]) for b in batch]
                max_len = max(len(s) for s in seqs)
                padded  = np.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
                mask    = np.zeros((len(seqs), max_len), dtype=np.int64)
                for i, s in enumerate(seqs):
                    padded[i, :len(s)] = s
                    mask[i, :len(s)]   = 1
                out[key]              = padded
                out[f"{key}_mask"]    = mask
            else:
                out[key] = default_collate([b[key] for b in batch])
        return out
    return _collate


# ─── Worker process for parallel loading ──────────────────────────────────────

def _worker_loop(dataset, index_queue, output_queue, collate_fn, worker_id, seed):
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    while True:
        item = index_queue.get()
        if item is None:
            output_queue.put(None)
            break
        batch_id, indices = item
        try:
            samples = [dataset[i] for i in indices]
            batch   = collate_fn(samples)
            output_queue.put((batch_id, batch, None))
        except Exception as e:
            output_queue.put((batch_id, None, str(e)))


# ─── DataLoader ───────────────────────────────────────────────────────────────

class DataLoader:
    """
    Multi-worker data loader with:
    - Configurable sampler/batch_sampler
    - Process-based parallelism (avoids GIL for CPU-bound preprocessing)
    - Pinned memory support for fast host→device transfer
    - Prefetching with bounded queue
    """

    def __init__(
        self,
        dataset:        Dataset,
        batch_size:     int = 1,
        shuffle:        bool = False,
        sampler:        Optional[Sampler] = None,
        batch_sampler:  Optional[Sampler] = None,
        num_workers:    int = 0,
        collate_fn:     Optional[Callable] = None,
        pin_memory:     bool = False,
        drop_last:      bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        seed:           int = 42,
    ):
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.collate_fn  = collate_fn or default_collate
        self.pin_memory  = pin_memory
        self.drop_last   = drop_last
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.seed        = seed

        if batch_sampler is not None:
            self.batch_sampler = batch_sampler
        else:
            base_sampler = sampler or (
                RandomSampler(dataset, seed=seed) if shuffle
                else SequentialSampler(dataset)
            )
            self.batch_sampler = BatchSampler(base_sampler, batch_size, drop_last)

        self._workers: List[mp.Process] = []
        self._index_q: Optional[mp.Queue]  = None
        self._output_q: Optional[mp.Queue] = None

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def __iter__(self) -> Iterator:
        if self.num_workers == 0:
            return self._single_process_iter()
        return self._multi_process_iter()

    # ── Single process (in-thread) iteration ───────────────────────────────

    def _single_process_iter(self):
        for batch_indices in self.batch_sampler:
            samples = [self.dataset[i] for i in batch_indices]
            batch   = self.collate_fn(samples)
            if self.pin_memory:
                batch = _pin_memory(batch)
            yield batch

    # ── Multi-process iteration with prefetch ──────────────────────────────

    def _multi_process_iter(self):
        index_q  = mp.Queue()
        output_q = mp.Queue(maxsize=self.num_workers * self.prefetch_factor)

        workers = [
            mp.Process(
                target=_worker_loop,
                args=(self.dataset, index_q, output_q, self.collate_fn, wid, self.seed),
                daemon=True,
            )
            for wid in range(self.num_workers)
        ]
        for w in workers: w.start()

        # Feed batch indices
        batches = list(enumerate(self.batch_sampler))
        for batch_id, indices in batches:
            index_q.put((batch_id, indices))
        for _ in range(self.num_workers):
            index_q.put(None)

        results = {}
        next_id = 0
        finished_workers = 0
        pending = len(batches)

        while pending > 0:
            batch_id, batch, err = output_q.get()
            if batch_id is None:
                finished_workers += 1
                continue
            if err:
                raise RuntimeError(f"DataLoader worker error: {err}")
            results[batch_id] = batch
            pending -= 1

            while next_id in results:
                out = results.pop(next_id)
                if self.pin_memory:
                    out = _pin_memory(out)
                yield out
                next_id += 1

        for w in workers: w.join(timeout=1.0)

    # ── Distributed-aware re-shuffling ──────────────────────────────────────

    def set_epoch(self, epoch: int):
        sampler = getattr(self.batch_sampler, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)


def _pin_memory(batch):
    """Pin host memory for faster async H2D copy (no-op in pure-Python sim)."""
    if isinstance(batch, dict):
        return {k: _pin_memory(v) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_pin_memory(x) for x in batch)
    return batch  # would call .pin_memory() on real Tensor


# ─── Infinite / Cycle iterator (for step-based training) ─────────────────────

class InfiniteDataLoader:
    """Wraps a DataLoader to cycle indefinitely, useful for max_steps training."""
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._iter  = iter(loader)
        self._epoch = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._epoch += 1
            self.loader.set_epoch(self._epoch)
            self._iter = iter(self.loader)
            return next(self._iter)


# ─── Prefetcher (async GPU prefetch via double-buffering) ────────────────────

class CUDAPrefetcher:
    """Overlaps H2D data transfer with compute using a background thread."""
    def __init__(self, loader: DataLoader, device):
        self.loader = loader
        self.device = device
        self._iter  = iter(loader)
        self._next_batch = None
        self._lock  = threading.Lock()
        self._preload()

    def _preload(self):
        try:
            batch = next(self._iter)
            self._next_batch = _to_device(batch, self.device)
        except StopIteration:
            self._next_batch = None

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            batch = self._next_batch
            if batch is None:
                raise StopIteration
            self._preload()
            return batch


def _to_device(batch, device):
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(x, device) for x in batch)
    if hasattr(batch, "to"):
        return batch.to(device, non_blocking=True)
    return batch
