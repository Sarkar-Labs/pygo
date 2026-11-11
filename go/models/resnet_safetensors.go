package models

import "time"

// GeneratedResnetSafetensors records execution state for the resnet safetensors component.
type GeneratedResnetSafetensors struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetSafetensors() *GeneratedResnetSafetensors {
	return &GeneratedResnetSafetensors{Name: "resnet_safetensors", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetSafetensors) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetSafetensors) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
