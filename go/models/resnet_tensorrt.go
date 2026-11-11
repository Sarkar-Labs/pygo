package models

import "time"

// GeneratedResnetTensorrt records execution state for the resnet tensorrt component.
type GeneratedResnetTensorrt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetTensorrt() *GeneratedResnetTensorrt {
	return &GeneratedResnetTensorrt{Name: "resnet_tensorrt", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetTensorrt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetTensorrt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
