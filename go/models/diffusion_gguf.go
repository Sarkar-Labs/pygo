package models

import "time"

// GeneratedDiffusionGguf records execution state for the diffusion gguf component.
type GeneratedDiffusionGguf struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDiffusionGguf() *GeneratedDiffusionGguf {
	return &GeneratedDiffusionGguf{Name: "diffusion_gguf", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDiffusionGguf) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDiffusionGguf) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
