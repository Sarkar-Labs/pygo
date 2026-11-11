package models

import "time"

// GeneratedMistralV4 records execution state for the mistral v4 component.
type GeneratedMistralV4 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistralV4() *GeneratedMistralV4 {
	return &GeneratedMistralV4{Name: "mistral_v4", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistralV4) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistralV4) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
