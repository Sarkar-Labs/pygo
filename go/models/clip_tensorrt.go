package models

import "time"

// GeneratedClipTensorrt records execution state for the clip tensorrt component.
type GeneratedClipTensorrt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipTensorrt() *GeneratedClipTensorrt {
	return &GeneratedClipTensorrt{Name: "clip_tensorrt", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipTensorrt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipTensorrt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
