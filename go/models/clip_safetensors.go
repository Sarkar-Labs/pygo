package models

import "time"

// GeneratedClipSafetensors records execution state for the clip safetensors component.
type GeneratedClipSafetensors struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedClipSafetensors() *GeneratedClipSafetensors {
	return &GeneratedClipSafetensors{Name: "clip_safetensors", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedClipSafetensors) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedClipSafetensors) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
