package models

import "time"

// GeneratedGptQuantized records execution state for the gpt quantized component.
type GeneratedGptQuantized struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptQuantized() *GeneratedGptQuantized {
	return &GeneratedGptQuantized{Name: "gpt_quantized", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptQuantized) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptQuantized) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
