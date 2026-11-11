package memory

import "sync/atomic"

// GeneratedFragmentCounter provides lock-free accounting for fragment.
type GeneratedFragmentCounter struct{ value atomic.Int64 }

func (c *GeneratedFragmentCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedFragmentCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedFragmentCounter) Reset()            { c.value.Store(0) }
