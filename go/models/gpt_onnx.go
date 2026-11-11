package models

import "time"

// GeneratedGptOnnx records execution state for the gpt onnx component.
type GeneratedGptOnnx struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptOnnx() *GeneratedGptOnnx {
	return &GeneratedGptOnnx{Name: "gpt_onnx", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptOnnx) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptOnnx) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
