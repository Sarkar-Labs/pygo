package models

import "time"

// GeneratedMobilenetTensorrt records execution state for the mobilenet tensorrt component.
type GeneratedMobilenetTensorrt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetTensorrt() *GeneratedMobilenetTensorrt {
	return &GeneratedMobilenetTensorrt{Name: "mobilenet_tensorrt", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetTensorrt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetTensorrt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
