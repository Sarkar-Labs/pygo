package models

import "time"

// GeneratedLlamaOpenvino records execution state for the llama openvino component.
type GeneratedLlamaOpenvino struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaOpenvino() *GeneratedLlamaOpenvino {
	return &GeneratedLlamaOpenvino{Name: "llama_openvino", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaOpenvino) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaOpenvino) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
