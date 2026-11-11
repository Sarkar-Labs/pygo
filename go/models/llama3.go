package models

import "time"

// GeneratedLlama3 records execution state for the llama3 component.
type GeneratedLlama3 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlama3() *GeneratedLlama3 {
	return &GeneratedLlama3{Name: "llama3", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlama3) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlama3) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
