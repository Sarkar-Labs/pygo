package data

import "time"

// GeneratedCache records execution state for the cache component.
type GeneratedCache struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCache() *GeneratedCache {
	return &GeneratedCache{Name: "cache", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCache) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCache) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
