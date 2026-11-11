package models

import "time"

// GeneratedLlamaDistilled records execution state for the llama distilled component.
type GeneratedLlamaDistilled struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLlamaDistilled() *GeneratedLlamaDistilled {
	return &GeneratedLlamaDistilled{Name: "llama_distilled", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLlamaDistilled) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLlamaDistilled) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
