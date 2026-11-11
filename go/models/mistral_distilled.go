package models

import "time"

// GeneratedMistralDistilled records execution state for the mistral distilled component.
type GeneratedMistralDistilled struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistralDistilled() *GeneratedMistralDistilled {
	return &GeneratedMistralDistilled{Name: "mistral_distilled", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistralDistilled) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistralDistilled) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
