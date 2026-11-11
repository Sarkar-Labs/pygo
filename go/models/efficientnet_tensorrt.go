package models

import "time"

// GeneratedEfficientnetTensorrt records execution state for the efficientnet tensorrt component.
type GeneratedEfficientnetTensorrt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEfficientnetTensorrt() *GeneratedEfficientnetTensorrt {
	return &GeneratedEfficientnetTensorrt{Name: "efficientnet_tensorrt", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEfficientnetTensorrt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEfficientnetTensorrt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
