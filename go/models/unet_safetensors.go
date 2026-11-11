package models

import "time"

// GeneratedUnetSafetensors records execution state for the unet safetensors component.
type GeneratedUnetSafetensors struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetSafetensors() *GeneratedUnetSafetensors {
	return &GeneratedUnetSafetensors{Name: "unet_safetensors", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetSafetensors) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetSafetensors) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
