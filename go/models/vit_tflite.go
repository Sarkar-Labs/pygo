package models

import "time"

// GeneratedVitTflite records execution state for the vit tflite component.
type GeneratedVitTflite struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitTflite() *GeneratedVitTflite {
	return &GeneratedVitTflite{Name: "vit_tflite", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitTflite) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitTflite) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
