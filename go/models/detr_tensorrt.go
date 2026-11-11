package models

import "time"

// GeneratedDetrTensorrt records execution state for the detr tensorrt component.
type GeneratedDetrTensorrt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrTensorrt() *GeneratedDetrTensorrt {
	return &GeneratedDetrTensorrt{Name: "detr_tensorrt", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrTensorrt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrTensorrt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
