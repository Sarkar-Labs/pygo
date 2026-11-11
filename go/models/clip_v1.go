package models

import "time"

// GeneratedClipV1 records execution state for the clip v1 component.
type GeneratedClipV1 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipV1() *GeneratedClipV1 {
	return &GeneratedClipV1{Name: "clip_v1", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipV1) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipV1) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
