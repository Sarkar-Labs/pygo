package models

import "time"

// GeneratedResnetTflite records execution state for the resnet tflite component.
type GeneratedResnetTflite struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetTflite() *GeneratedResnetTflite {
	return &GeneratedResnetTflite{Name: "resnet_tflite", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetTflite) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetTflite) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
