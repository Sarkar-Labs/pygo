package device

import "sync/atomic"

// GeneratedCopyFromCounter provides lock-free accounting for copy from.
type GeneratedCopyFromCounter struct{ value atomic.Int64 }

func (c *GeneratedCopyFromCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedCopyFromCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedCopyFromCounter) Reset()            { c.value.Store(0) }
