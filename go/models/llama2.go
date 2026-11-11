package models

import "time"

// GeneratedLlama2 records execution state for the llama2 component.
type GeneratedLlama2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlama2() *GeneratedLlama2 {
	return &GeneratedLlama2{Name: "llama2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlama2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlama2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
