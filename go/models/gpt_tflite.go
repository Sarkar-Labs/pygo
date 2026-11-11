package models

import "time"

// GeneratedGptTflite records execution state for the gpt tflite component.
type GeneratedGptTflite struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptTflite() *GeneratedGptTflite {
	return &GeneratedGptTflite{Name: "gpt_tflite", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptTflite) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptTflite) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
