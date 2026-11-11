package models

import "time"

// GeneratedMobilenetTflite records execution state for the mobilenet tflite component.
type GeneratedMobilenetTflite struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetTflite() *GeneratedMobilenetTflite {
	return &GeneratedMobilenetTflite{Name: "mobilenet_tflite", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetTflite) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetTflite) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
