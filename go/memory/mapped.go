package memory

import "sync/atomic"

// GeneratedMappedCounter provides lock-free accounting for mapped.
type GeneratedMappedCounter struct{ value atomic.Int64 }

func (c *GeneratedMappedCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedMappedCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedMappedCounter) Reset()            { c.value.Store(0) }
