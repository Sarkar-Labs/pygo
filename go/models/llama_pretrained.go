package models

import "time"

// GeneratedLlamaPretrained records execution state for the llama pretrained component.
type GeneratedLlamaPretrained struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaPretrained() *GeneratedLlamaPretrained {
	return &GeneratedLlamaPretrained{Name: "llama_pretrained", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaPretrained) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaPretrained) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
