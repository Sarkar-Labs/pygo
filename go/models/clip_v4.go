package models

import "time"

// GeneratedClipV4 records execution state for the clip v4 component.
type GeneratedClipV4 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipV4() *GeneratedClipV4 {
	return &GeneratedClipV4{Name: "clip_v4", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipV4) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipV4) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
