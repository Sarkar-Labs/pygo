package models

import "time"

// GeneratedYoloSafetensors records execution state for the yolo safetensors component.
type GeneratedYoloSafetensors struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloSafetensors() *GeneratedYoloSafetensors {
	return &GeneratedYoloSafetensors{Name: "yolo_safetensors", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloSafetensors) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloSafetensors) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
