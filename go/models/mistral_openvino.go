package models

import "time"

// GeneratedMistralOpenvino records execution state for the mistral openvino component.
type GeneratedMistralOpenvino struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistralOpenvino() *GeneratedMistralOpenvino {
	return &GeneratedMistralOpenvino{Name: "mistral_openvino", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistralOpenvino) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistralOpenvino) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
