package models

import "time"

// GeneratedVitGguf records execution state for the vit gguf component.
type GeneratedVitGguf struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitGguf() *GeneratedVitGguf {
	return &GeneratedVitGguf{Name: "vit_gguf", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitGguf) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitGguf) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
