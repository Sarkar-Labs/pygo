package core

import "sync/atomic"

// GeneratedBuilderCounter provides lock-free accounting for builder.
type GeneratedBuilderCounter struct{ value atomic.Int64 }

func (c *GeneratedBuilderCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedBuilderCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedBuilderCounter) Reset()            { c.value.Store(0) }
