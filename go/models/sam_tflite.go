package models

import "time"

// GeneratedSamTflite records execution state for the sam tflite component.
type GeneratedSamTflite struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamTflite() *GeneratedSamTflite {
	return &GeneratedSamTflite{Name: "sam_tflite", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamTflite) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamTflite) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
