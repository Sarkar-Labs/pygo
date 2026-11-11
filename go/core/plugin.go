package core

import "sync/atomic"

// GeneratedPluginCounter provides lock-free accounting for plugin.
type GeneratedPluginCounter struct{ value atomic.Int64 }

func (c *GeneratedPluginCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedPluginCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedPluginCounter) Reset()            { c.value.Store(0) }
