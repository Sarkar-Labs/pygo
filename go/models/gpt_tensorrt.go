package models

import "time"

// GeneratedGptTensorrt records execution state for the gpt tensorrt component.
type GeneratedGptTensorrt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptTensorrt() *GeneratedGptTensorrt {
	return &GeneratedGptTensorrt{Name: "gpt_tensorrt", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptTensorrt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptTensorrt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
