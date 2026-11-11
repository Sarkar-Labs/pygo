package models

import "time"

// GeneratedMistralV2 records execution state for the mistral v2 component.
type GeneratedMistralV2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistralV2() *GeneratedMistralV2 {
	return &GeneratedMistralV2{Name: "mistral_v2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistralV2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistralV2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
