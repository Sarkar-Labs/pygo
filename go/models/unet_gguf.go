package models

import "time"

// GeneratedUnetGguf records execution state for the unet gguf component.
type GeneratedUnetGguf struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetGguf() *GeneratedUnetGguf {
	return &GeneratedUnetGguf{Name: "unet_gguf", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetGguf) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetGguf) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
