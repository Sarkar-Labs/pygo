package models

import "time"

// GeneratedClipGguf records execution state for the clip gguf component.
type GeneratedClipGguf struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipGguf() *GeneratedClipGguf {
	return &GeneratedClipGguf{Name: "clip_gguf", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipGguf) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipGguf) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
