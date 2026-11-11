package device

import "sync/atomic"

// GeneratedRuntimeVersionCounter provides lock-free accounting for runtime version.
type GeneratedRuntimeVersionCounter struct{ value atomic.Int64 }

func (c *GeneratedRuntimeVersionCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedRuntimeVersionCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedRuntimeVersionCounter) Reset()            { c.value.Store(0) }
