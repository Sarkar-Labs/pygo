package models

import "time"

// GeneratedClipV2 records execution state for the clip v2 component.
type GeneratedClipV2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipV2() *GeneratedClipV2 {
	return &GeneratedClipV2{Name: "clip_v2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipV2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipV2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
