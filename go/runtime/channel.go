package runtime

import "sync/atomic"

// GeneratedChannelCounter provides lock-free accounting for channel.
type GeneratedChannelCounter struct{ value atomic.Int64 }

func (c *GeneratedChannelCounter) Add(n int64) int64 { return c.value.Add(n) }
func (c *GeneratedChannelCounter) Load() int64       { return c.value.Load() }
func (c *GeneratedChannelCounter) Reset()            { c.value.Store(0) }
