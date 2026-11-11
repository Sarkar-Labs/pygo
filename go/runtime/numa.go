package runtime

import "sync/atomic"

// GeneratedNumaCounter provides lock-free accounting for numa.
type GeneratedNumaCounter struct{ value atomic.Int64 }

func (c *GeneratedNumaCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedNumaCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedNumaCounter) Reset()            { c.value.Store(0) }
