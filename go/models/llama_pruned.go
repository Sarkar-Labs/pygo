package models

import "time"

// GeneratedLlamaPruned records execution state for the llama pruned component.
type GeneratedLlamaPruned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaPruned() *GeneratedLlamaPruned {
	return &GeneratedLlamaPruned{Name: "llama_pruned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaPruned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaPruned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
