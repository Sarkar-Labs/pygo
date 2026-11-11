package models

import "time"

// GeneratedMobilenetSafetensors records execution state for the mobilenet safetensors component.
type GeneratedMobilenetSafetensors struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetSafetensors() *GeneratedMobilenetSafetensors {
	return &GeneratedMobilenetSafetensors{Name: "mobilenet_safetensors", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetSafetensors) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetSafetensors) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
