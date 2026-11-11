package device

import "sync/atomic"

// GeneratedCopyToCounter provides lock-free accounting for copy to.
type GeneratedCopyToCounter struct{ value atomic.Int64 }

func (c *GeneratedCopyToCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedCopyToCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedCopyToCounter) Reset()            { c.value.Store(0) }
