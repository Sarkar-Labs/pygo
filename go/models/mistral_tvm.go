package models

import "time"

// GeneratedMistralTvm records execution state for the mistral tvm component.
type GeneratedMistralTvm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistralTvm() *GeneratedMistralTvm {
	return &GeneratedMistralTvm{Name: "mistral_tvm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistralTvm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistralTvm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
