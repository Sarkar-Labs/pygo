package models

import "time"

// GeneratedLlamaTvm records execution state for the llama tvm component.
type GeneratedLlamaTvm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaTvm() *GeneratedLlamaTvm {
	return &GeneratedLlamaTvm{Name: "llama_tvm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaTvm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaTvm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
