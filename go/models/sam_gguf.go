package models

import "time"

// GeneratedSamGguf records execution state for the sam gguf component.
type GeneratedSamGguf struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamGguf() *GeneratedSamGguf {
	return &GeneratedSamGguf{Name: "sam_gguf", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamGguf) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamGguf) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
