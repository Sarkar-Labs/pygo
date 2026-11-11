package models

import "time"

// GeneratedMistralV1 records execution state for the mistral v1 component.
type GeneratedMistralV1 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistralV1() *GeneratedMistralV1 {
	return &GeneratedMistralV1{Name: "mistral_v1", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistralV1) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistralV1) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
