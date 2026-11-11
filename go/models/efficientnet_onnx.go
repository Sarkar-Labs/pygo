package models

import "time"

// GeneratedEfficientnetOnnx records execution state for the efficientnet onnx component.
type GeneratedEfficientnetOnnx struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEfficientnetOnnx() *GeneratedEfficientnetOnnx {
	return &GeneratedEfficientnetOnnx{Name: "efficientnet_onnx", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEfficientnetOnnx) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEfficientnetOnnx) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
