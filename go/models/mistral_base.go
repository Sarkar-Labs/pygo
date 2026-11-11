package models

import "time"

// GeneratedMistralBase records execution state for the mistral base component.
type GeneratedMistralBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistralBase() *GeneratedMistralBase {
	return &GeneratedMistralBase{Name: "mistral_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistralBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistralBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
