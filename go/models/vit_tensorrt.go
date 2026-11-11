package models

import "time"

// GeneratedVitTensorrt records execution state for the vit tensorrt component.
type GeneratedVitTensorrt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitTensorrt() *GeneratedVitTensorrt {
	return &GeneratedVitTensorrt{Name: "vit_tensorrt", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitTensorrt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitTensorrt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
