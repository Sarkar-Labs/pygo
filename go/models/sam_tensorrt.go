package models

import "time"

// GeneratedSamTensorrt records execution state for the sam tensorrt component.
type GeneratedSamTensorrt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamTensorrt() *GeneratedSamTensorrt {
	return &GeneratedSamTensorrt{Name: "sam_tensorrt", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamTensorrt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamTensorrt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
