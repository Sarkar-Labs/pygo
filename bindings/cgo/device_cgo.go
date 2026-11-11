// device_cgo.go — Device management and memory allocation via CGo.
package cgo

/*
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

// ─── CPU memory ───────────────────────────────────────────────────────────────

static void* pygo_cpu_malloc(int64_t nbytes) {
    void* ptr = malloc((size_t)nbytes);
    if (ptr) memset(ptr, 0, (size_t)nbytes);
    return ptr;
}

static void* pygo_cpu_malloc_aligned(int64_t nbytes, int alignment) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, (size_t)alignment, (size_t)nbytes) != 0) return NULL;
    memset(ptr, 0, (size_t)nbytes);
    return ptr;
}

static void pygo_cpu_free(void* ptr) { if (ptr) free(ptr); }

static void pygo_cpu_memcpy(void* dst, const void* src, int64_t nbytes) {
    memcpy(dst, src, (size_t)nbytes);
}

static void pygo_cpu_memset(void* ptr, int val, int64_t nbytes) {
    memset(ptr, val, (size_t)nbytes);
}

// ─── Device info (CPU) ────────────────────────────────────────────────────────

static int pygo_cpu_count() { return 1; }

static int64_t pygo_cpu_total_memory() {
    // Read from /proc/meminfo on Linux
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    int64_t total = -1;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        int64_t val;
        if (sscanf(line, "MemTotal: %ld kB", &val) == 1) {
            total = val * 1024LL; break;
        }
    }
    fclose(f);
    return total;
}

static int64_t pygo_cpu_free_memory() {
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    int64_t free_mem = -1;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        int64_t val;
        if (sscanf(line, "MemAvailable: %ld kB", &val) == 1) {
            free_mem = val * 1024LL; break;
        }
    }
    fclose(f);
    return free_mem;
}
*/
import "C"
import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// ─── Device kinds ─────────────────────────────────────────────────────────────

// DeviceKind represents compute device type.
type DeviceKind int

const (
	DevCPU    DeviceKind = 0
	DevCUDA   DeviceKind = 1
	DevMetal  DeviceKind = 2
	DevVulkan DeviceKind = 3
)

func (d DeviceKind) String() string {
	return [...]string{"cpu", "cuda", "metal", "vulkan"}[d]
}

// Device represents a specific compute device.
type Device struct {
	Kind  DeviceKind
	Index int
}

var (
	CPU   = Device{DevCPU, 0}
	CUDA0 = Device{DevCUDA, 0}
	CUDA1 = Device{DevCUDA, 1}
)

func CUDA(index int) Device { return Device{DevCUDA, index} }

func (d Device) String() string {
	if d.Kind == DevCPU {
		return "cpu"
	}
	return fmt.Sprintf("%s:%d", d.Kind, d.Index)
}

func (d Device) IsCPU() bool  { return d.Kind == DevCPU }
func (d Device) IsCUDA() bool { return d.Kind == DevCUDA }

// ─── DeviceInfo ───────────────────────────────────────────────────────────────

// CPUInfo holds CPU device properties.
type CPUInfo struct {
	NumCores    int
	TotalMemory int64
	FreeMemory  int64
	GoMaxProcs  int
}

// GetCPUInfo returns current CPU device properties.
func GetCPUInfo() CPUInfo {
	return CPUInfo{
		NumCores:    int(C.pygo_cpu_count()),
		TotalMemory: int64(C.pygo_cpu_total_memory()),
		FreeMemory:  int64(C.pygo_cpu_free_memory()),
		GoMaxProcs:  runtime.GOMAXPROCS(0),
	}
}

func (c CPUInfo) String() string {
	return fmt.Sprintf("CPU(cores=%d total=%.2fGB free=%.2fGB gomaxprocs=%d)",
		c.NumCores,
		float64(c.TotalMemory)/1e9,
		float64(c.FreeMemory)/1e9,
		c.GoMaxProcs,
	)
}

// ─── Memory allocator ─────────────────────────────────────────────────────────

// AllocResult holds an allocation result.
type AllocResult struct {
	Ptr    unsafe.Pointer
	NBytes int64
	Device Device
}

// CPUAlloc allocates aligned CPU memory (64-byte aligned for AVX-512).
func CPUAlloc(nbytes int64) (*AllocResult, error) {
	if nbytes <= 0 {
		return nil, fmt.Errorf("CPUAlloc: invalid size %d", nbytes)
	}
	ptr := C.pygo_cpu_malloc_aligned(C.int64_t(nbytes), 64)
	if ptr == nil {
		return nil, fmt.Errorf("CPUAlloc: malloc failed for %d bytes", nbytes)
	}
	return &AllocResult{Ptr: unsafe.Pointer(ptr), NBytes: nbytes, Device: CPU}, nil
}

// Free releases a CPU allocation.
func (a *AllocResult) Free() {
	if a != nil && a.Ptr != nil {
		C.pygo_cpu_free(a.Ptr)
		a.Ptr = nil
	}
}

// ─── Memory pool ─────────────────────────────────────────────────────────────

// MemPool is a simple size-bucketed free list.
type MemPool struct {
	mu    sync.Mutex
	free  map[int64][]unsafe.Pointer
	stats PoolStats
}

// PoolStats tracks allocator statistics.
type PoolStats struct {
	TotalAllocs int64
	TotalFrees  int64
	AllocBytes  int64
	PeakBytes   int64
	PoolHits    int64
	PoolMisses  int64
}

// NewMemPool creates a new memory pool.
func NewMemPool() *MemPool {
	return &MemPool{free: make(map[int64][]unsafe.Pointer)}
}

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

// Alloc returns a pointer of at least nbytes, from pool if available.
func (p *MemPool) Alloc(nbytes int64) unsafe.Pointer {
	sz := nextPow2(nbytes)
	p.mu.Lock()
	if pool := p.free[sz]; len(pool) > 0 {
		ptr := pool[len(pool)-1]
		p.free[sz] = pool[:len(pool)-1]
		p.stats.PoolHits++
		p.stats.AllocBytes += sz
		p.stats.TotalAllocs++
		if p.stats.AllocBytes > p.stats.PeakBytes {
			p.stats.PeakBytes = p.stats.AllocBytes
		}
		p.mu.Unlock()
		return ptr
	}
	p.stats.PoolMisses++
	p.stats.TotalAllocs++
	p.stats.AllocBytes += sz
	if p.stats.AllocBytes > p.stats.PeakBytes {
		p.stats.PeakBytes = p.stats.AllocBytes
	}
	p.mu.Unlock()
	return unsafe.Pointer(C.pygo_cpu_malloc_aligned(C.int64_t(sz), 64))
}

// Free returns memory to the pool.
func (p *MemPool) Free(ptr unsafe.Pointer, nbytes int64) {
	if ptr == nil {
		return
	}
	sz := nextPow2(nbytes)
	p.mu.Lock()
	p.free[sz] = append(p.free[sz], ptr)
	p.stats.TotalFrees++
	p.stats.AllocBytes -= sz
	p.mu.Unlock()
}

// Stats returns pool statistics.
func (p *MemPool) Stats() PoolStats { p.mu.Lock(); defer p.mu.Unlock(); return p.stats }

// Summary returns a human-readable pool summary.
func (p *MemPool) Summary() string {
	s := p.Stats()
	return fmt.Sprintf(
		"MemPool(alloc=%.2fMB peak=%.2fMB hits=%d misses=%d allocs=%d frees=%d)",
		float64(s.AllocBytes)/1e6, float64(s.PeakBytes)/1e6,
		s.PoolHits, s.PoolMisses, s.TotalAllocs, s.TotalFrees,
	)
}

// Global default pool
var DefaultPool = NewMemPool()

// ─── Memcpy / Memset ─────────────────────────────────────────────────────────

// Memcpy copies nbytes from src to dst.
func Memcpy(dst, src unsafe.Pointer, nbytes int64) {
	C.pygo_cpu_memcpy(dst, src, C.int64_t(nbytes))
}

// Memset fills nbytes starting at ptr with val.
func Memset(ptr unsafe.Pointer, val int, nbytes int64) {
	C.pygo_cpu_memset(ptr, C.int(val), C.int64_t(nbytes))
}
