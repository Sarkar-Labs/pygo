package models

import "time"

// GeneratedMobilenetOnnx records execution state for the mobilenet onnx component.
type GeneratedMobilenetOnnx struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetOnnx() *GeneratedMobilenetOnnx {
	return &GeneratedMobilenetOnnx{Name: "mobilenet_onnx", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetOnnx) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetOnnx) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
