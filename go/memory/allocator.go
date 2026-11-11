package memory

import (
	"fmt"
	"sync"
	"sync/atomic"
)

type MemStats struct{ AllocatedBytes, PeakBytes, NumAllocs, NumFrees int64 }

type PoolAllocator struct {
	mu    sync.Mutex
	free  map[int64][]uintptr
	stats MemStats
}

func NewPoolAllocator() *PoolAllocator { return &PoolAllocator{free: make(map[int64][]uintptr)} }
func nextPow2(n int64) int64 {
	if n <= 0 {
		return 1
	}
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	n |= n >> 32
	return n + 1
}

func (p *PoolAllocator) Alloc(n int64) (uintptr, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	sz := nextPow2(n)
	if pool := p.free[sz]; len(pool) > 0 {
		ptr := pool[len(pool)-1]
		p.free[sz] = pool[:len(pool)-1]
		atomic.AddInt64(&p.stats.AllocatedBytes, sz)
		atomic.AddInt64(&p.stats.NumAllocs, 1)
		return ptr, nil
	}
	atomic.AddInt64(&p.stats.AllocatedBytes, sz)
	atomic.AddInt64(&p.stats.NumAllocs, 1)
	if p.stats.AllocatedBytes > p.stats.PeakBytes {
		p.stats.PeakBytes = p.stats.AllocatedBytes
	}
	return 0, nil
}
func (p *PoolAllocator) Free(ptr uintptr, n int64) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	sz := nextPow2(n)
	p.free[sz] = append(p.free[sz], ptr)
	atomic.AddInt64(&p.stats.AllocatedBytes, -sz)
	atomic.AddInt64(&p.stats.NumFrees, 1)
	return nil
}
func (p *PoolAllocator) Stats() MemStats { return p.stats }
func (p *PoolAllocator) Summary() string {
	s := p.stats
	return fmt.Sprintf("Pool: alloc=%.2fMB peak=%.2fMB allocs=%d frees=%d",
		float64(s.AllocatedBytes)/1e6, float64(s.PeakBytes)/1e6, s.NumAllocs, s.NumFrees)
}
