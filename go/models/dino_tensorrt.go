package models

import "time"

// GeneratedDinoTensorrt records execution state for the dino tensorrt component.
type GeneratedDinoTensorrt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoTensorrt() *GeneratedDinoTensorrt {
	return &GeneratedDinoTensorrt{Name: "dino_tensorrt", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoTensorrt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoTensorrt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
