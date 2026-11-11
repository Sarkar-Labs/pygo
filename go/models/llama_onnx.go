package models

import "time"

// GeneratedLlamaOnnx records execution state for the llama onnx component.
type GeneratedLlamaOnnx struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaOnnx() *GeneratedLlamaOnnx {
	return &GeneratedLlamaOnnx{Name: "llama_onnx", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaOnnx) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaOnnx) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
