package models

import "time"

// GeneratedMistralPruned records execution state for the mistral pruned component.
type GeneratedMistralPruned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistralPruned() *GeneratedMistralPruned {
	return &GeneratedMistralPruned{Name: "mistral_pruned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistralPruned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistralPruned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
