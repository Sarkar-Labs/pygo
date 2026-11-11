package models

import "time"

// GeneratedMistralLarge records execution state for the mistral large component.
type GeneratedMistralLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistralLarge() *GeneratedMistralLarge {
	return &GeneratedMistralLarge{Name: "mistral_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistralLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistralLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
