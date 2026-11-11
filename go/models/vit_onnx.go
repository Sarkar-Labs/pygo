package models

import "time"

// GeneratedVitOnnx records execution state for the vit onnx component.
type GeneratedVitOnnx struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitOnnx() *GeneratedVitOnnx {
	return &GeneratedVitOnnx{Name: "vit_onnx", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitOnnx) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitOnnx) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
