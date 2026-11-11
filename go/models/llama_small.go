package models

import "time"

// GeneratedLlamaSmall records execution state for the llama small component.
type GeneratedLlamaSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaSmall() *GeneratedLlamaSmall {
	return &GeneratedLlamaSmall{Name: "llama_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
