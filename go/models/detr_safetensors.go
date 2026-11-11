package models

import "time"

// GeneratedDetrSafetensors records execution state for the detr safetensors component.
type GeneratedDetrSafetensors struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrSafetensors() *GeneratedDetrSafetensors {
	return &GeneratedDetrSafetensors{Name: "detr_safetensors", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrSafetensors) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrSafetensors) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
