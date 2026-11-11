package models

import "time"

// GeneratedUnetOnnx records execution state for the unet onnx component.
type GeneratedUnetOnnx struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetOnnx() *GeneratedUnetOnnx {
	return &GeneratedUnetOnnx{Name: "unet_onnx", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetOnnx) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetOnnx) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
