package models

import "time"

// GeneratedClipTflite records execution state for the clip tflite component.
type GeneratedClipTflite struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipTflite() *GeneratedClipTflite {
	return &GeneratedClipTflite{Name: "clip_tflite", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipTflite) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipTflite) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
