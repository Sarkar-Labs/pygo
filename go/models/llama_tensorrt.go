package models

import "time"

// GeneratedLlamaTensorrt records execution state for the llama tensorrt component.
type GeneratedLlamaTensorrt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaTensorrt() *GeneratedLlamaTensorrt {
	return &GeneratedLlamaTensorrt{Name: "llama_tensorrt", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaTensorrt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaTensorrt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
