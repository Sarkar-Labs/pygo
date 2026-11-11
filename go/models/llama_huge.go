package models

import "time"

// GeneratedLlamaHuge records execution state for the llama huge component.
type GeneratedLlamaHuge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaHuge() *GeneratedLlamaHuge {
	return &GeneratedLlamaHuge{Name: "llama_huge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaHuge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaHuge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
