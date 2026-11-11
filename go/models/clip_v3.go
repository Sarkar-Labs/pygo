package models

import "time"

// GeneratedClipV3 records execution state for the clip v3 component.
type GeneratedClipV3 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipV3() *GeneratedClipV3 {
	return &GeneratedClipV3{Name: "clip_v3", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipV3) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipV3) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
