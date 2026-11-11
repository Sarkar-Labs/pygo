package models

import "time"

// GeneratedMistralPretrained records execution state for the mistral pretrained component.
type GeneratedMistralPretrained struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistralPretrained() *GeneratedMistralPretrained {
	return &GeneratedMistralPretrained{Name: "mistral_pretrained", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistralPretrained) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistralPretrained) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
