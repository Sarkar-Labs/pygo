package models

import "time"

// GeneratedResnetGguf records execution state for the resnet gguf component.
type GeneratedResnetGguf struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetGguf() *GeneratedResnetGguf {
	return &GeneratedResnetGguf{Name: "resnet_gguf", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetGguf) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetGguf) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
