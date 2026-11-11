package models

import "time"

// GeneratedDetrOnnx records execution state for the detr onnx component.
type GeneratedDetrOnnx struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrOnnx() *GeneratedDetrOnnx {
	return &GeneratedDetrOnnx{Name: "detr_onnx", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrOnnx) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrOnnx) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
