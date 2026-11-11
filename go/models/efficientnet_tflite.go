package models

import "time"

// GeneratedEfficientnetTflite records execution state for the efficientnet tflite component.
type GeneratedEfficientnetTflite struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEfficientnetTflite() *GeneratedEfficientnetTflite {
	return &GeneratedEfficientnetTflite{Name: "efficientnet_tflite", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEfficientnetTflite) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEfficientnetTflite) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
