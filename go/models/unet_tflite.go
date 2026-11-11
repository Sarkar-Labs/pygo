package models

import "time"

// GeneratedUnetTflite records execution state for the unet tflite component.
type GeneratedUnetTflite struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetTflite() *GeneratedUnetTflite {
	return &GeneratedUnetTflite{Name: "unet_tflite", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetTflite) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetTflite) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
