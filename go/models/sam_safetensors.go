package models

import "time"

// GeneratedSamSafetensors records execution state for the sam safetensors component.
type GeneratedSamSafetensors struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamSafetensors() *GeneratedSamSafetensors {
	return &GeneratedSamSafetensors{Name: "sam_safetensors", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamSafetensors) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamSafetensors) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
