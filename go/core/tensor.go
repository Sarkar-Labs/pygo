package core

import "sync/atomic"

// GeneratedTensorCounter provides lock-free accounting for tensor.
type GeneratedTensorCounter struct{ value atomic.Int64 }

func (c *GeneratedTensorCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedTensorCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedTensorCounter) Reset()            { c.value.Store(0) }
