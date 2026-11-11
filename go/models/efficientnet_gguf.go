package models

import "time"

// GeneratedEfficientnetGguf records execution state for the efficientnet gguf component.
type GeneratedEfficientnetGguf struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEfficientnetGguf() *GeneratedEfficientnetGguf {
	return &GeneratedEfficientnetGguf{Name: "efficientnet_gguf", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEfficientnetGguf) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEfficientnetGguf) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
