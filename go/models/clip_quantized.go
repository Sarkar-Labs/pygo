package models

import "time"

// GeneratedClipQuantized records execution state for the clip quantized component.
type GeneratedClipQuantized struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipQuantized() *GeneratedClipQuantized {
	return &GeneratedClipQuantized{Name: "clip_quantized", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipQuantized) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipQuantized) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
