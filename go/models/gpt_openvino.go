package models

import "time"

// GeneratedGptOpenvino records execution state for the gpt openvino component.
type GeneratedGptOpenvino struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptOpenvino() *GeneratedGptOpenvino {
	return &GeneratedGptOpenvino{Name: "gpt_openvino", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptOpenvino) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptOpenvino) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
