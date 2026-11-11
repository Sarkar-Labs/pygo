package models

import "time"

// GeneratedMistralTflite records execution state for the mistral tflite component.
type GeneratedMistralTflite struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistralTflite() *GeneratedMistralTflite {
	return &GeneratedMistralTflite{Name: "mistral_tflite", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistralTflite) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistralTflite) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
