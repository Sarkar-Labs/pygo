package models

import "time"

// GeneratedDetrTflite records execution state for the detr tflite component.
type GeneratedDetrTflite struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrTflite() *GeneratedDetrTflite {
	return &GeneratedDetrTflite{Name: "detr_tflite", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrTflite) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrTflite) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
