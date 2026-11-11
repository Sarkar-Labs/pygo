package core

import "sync/atomic"

// GeneratedFactoryCounter provides lock-free accounting for factory.
type GeneratedFactoryCounter struct{ value atomic.Int64 }

func (c *GeneratedFactoryCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedFactoryCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedFactoryCounter) Reset()            { c.value.Store(0) }
