package models

import "time"

// GeneratedClipOnnx records execution state for the clip onnx component.
type GeneratedClipOnnx struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipOnnx() *GeneratedClipOnnx {
	return &GeneratedClipOnnx{Name: "clip_onnx", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipOnnx) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipOnnx) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
