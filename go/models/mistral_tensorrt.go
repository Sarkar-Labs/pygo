package models

import "time"

// GeneratedMistralTensorrt records execution state for the mistral tensorrt component.
type GeneratedMistralTensorrt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistralTensorrt() *GeneratedMistralTensorrt {
	return &GeneratedMistralTensorrt{Name: "mistral_tensorrt", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistralTensorrt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistralTensorrt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
