package models

import "time"

// GeneratedEfficientnetSafetensors records execution state for the efficientnet safetensors component.
type GeneratedEfficientnetSafetensors struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEfficientnetSafetensors() *GeneratedEfficientnetSafetensors {
	return &GeneratedEfficientnetSafetensors{Name: "efficientnet_safetensors", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEfficientnetSafetensors) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEfficientnetSafetensors) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
