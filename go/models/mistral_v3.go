package models

import "time"

// GeneratedMistralV3 records execution state for the mistral v3 component.
type GeneratedMistralV3 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistralV3() *GeneratedMistralV3 {
	return &GeneratedMistralV3{Name: "mistral_v3", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistralV3) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistralV3) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
