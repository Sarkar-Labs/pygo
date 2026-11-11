package models

import "time"

// GeneratedUnetTensorrt records execution state for the unet tensorrt component.
type GeneratedUnetTensorrt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetTensorrt() *GeneratedUnetTensorrt {
	return &GeneratedUnetTensorrt{Name: "unet_tensorrt", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetTensorrt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetTensorrt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
