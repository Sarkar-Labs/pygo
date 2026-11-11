package models

import "time"

// GeneratedClipVit records execution state for the clip vit component.
type GeneratedClipVit struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipVit() *GeneratedClipVit {
	return &GeneratedClipVit{Name: "clip_vit", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipVit) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipVit) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
