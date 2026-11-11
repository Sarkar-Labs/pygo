package device

import "sync/atomic"

// GeneratedSynchronizeCounter provides lock-free accounting for synchronize.
type GeneratedSynchronizeCounter struct{ value atomic.Int64 }

func (c *GeneratedSynchronizeCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedSynchronizeCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedSynchronizeCounter) Reset()            { c.value.Store(0) }
