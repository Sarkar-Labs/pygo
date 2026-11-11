package models

import "time"

// GeneratedClipLarge records execution state for the clip large component.
type GeneratedClipLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipLarge() *GeneratedClipLarge {
	return &GeneratedClipLarge{Name: "clip_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
