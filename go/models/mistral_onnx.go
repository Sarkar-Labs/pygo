package models

import "time"

// GeneratedMistralOnnx records execution state for the mistral onnx component.
type GeneratedMistralOnnx struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistralOnnx() *GeneratedMistralOnnx {
	return &GeneratedMistralOnnx{Name: "mistral_onnx", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistralOnnx) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistralOnnx) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
