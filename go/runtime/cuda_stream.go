package runtime

import "sync/atomic"

// GeneratedCudaStreamCounter provides lock-free accounting for cuda stream.
type GeneratedCudaStreamCounter struct{ value atomic.Int64 }

func (c *GeneratedCudaStreamCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedCudaStreamCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedCudaStreamCounter) Reset()            { c.value.Store(0) }
