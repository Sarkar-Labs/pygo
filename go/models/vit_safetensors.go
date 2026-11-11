package models

import "time"

// GeneratedVitSafetensors records execution state for the vit safetensors component.
type GeneratedVitSafetensors struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitSafetensors() *GeneratedVitSafetensors {
	return &GeneratedVitSafetensors{Name: "vit_safetensors", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitSafetensors) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitSafetensors) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
