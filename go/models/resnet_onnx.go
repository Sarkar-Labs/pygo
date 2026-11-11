package models

import "time"

// GeneratedResnetOnnx records execution state for the resnet onnx component.
type GeneratedResnetOnnx struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetOnnx() *GeneratedResnetOnnx {
	return &GeneratedResnetOnnx{Name: "resnet_onnx", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetOnnx) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetOnnx) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
