package runtime

import "sync/atomic"

// GeneratedAsyncExecutorCounter provides lock-free accounting for async executor.
type GeneratedAsyncExecutorCounter struct{ value atomic.Int64 }

func (c *GeneratedAsyncExecutorCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedAsyncExecutorCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedAsyncExecutorCounter) Reset()            { c.value.Store(0) }
