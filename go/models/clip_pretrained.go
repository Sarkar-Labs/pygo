package models

import "time"

// GeneratedClipPretrained records execution state for the clip pretrained component.
type GeneratedClipPretrained struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipPretrained() *GeneratedClipPretrained {
	return &GeneratedClipPretrained{Name: "clip_pretrained", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipPretrained) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipPretrained) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
