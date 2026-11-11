package models

import "time"

// GeneratedLlamaTflite records execution state for the llama tflite component.
type GeneratedLlamaTflite struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaTflite() *GeneratedLlamaTflite {
	return &GeneratedLlamaTflite{Name: "llama_tflite", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaTflite) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaTflite) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
