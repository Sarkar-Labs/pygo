package runtime

import "sync/atomic"

// GeneratedWorkStealingCounter provides lock-free accounting for work stealing.
type GeneratedWorkStealingCounter struct{ value atomic.Int64 }

func (c *GeneratedWorkStealingCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedWorkStealingCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedWorkStealingCounter) Reset()            { c.value.Store(0) }
