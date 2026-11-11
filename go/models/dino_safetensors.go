package models

import "time"

// GeneratedDinoSafetensors records execution state for the dino safetensors component.
type GeneratedDinoSafetensors struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoSafetensors() *GeneratedDinoSafetensors {
	return &GeneratedDinoSafetensors{Name: "dino_safetensors", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoSafetensors) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoSafetensors) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
