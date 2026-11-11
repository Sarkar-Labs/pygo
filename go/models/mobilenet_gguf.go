package models

import "time"

// GeneratedMobilenetGguf records execution state for the mobilenet gguf component.
type GeneratedMobilenetGguf struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMobilenetGguf() *GeneratedMobilenetGguf {
	return &GeneratedMobilenetGguf{Name: "mobilenet_gguf", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMobilenetGguf) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMobilenetGguf) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
