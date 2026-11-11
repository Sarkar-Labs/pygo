package runtime

import "sync/atomic"

// GeneratedAffinityCounter provides lock-free accounting for affinity.
type GeneratedAffinityCounter struct{ value atomic.Int64 }

func (c *GeneratedAffinityCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedAffinityCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedAffinityCounter) Reset()            { c.value.Store(0) }
