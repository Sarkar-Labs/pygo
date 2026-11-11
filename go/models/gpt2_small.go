package models

import "time"

// GeneratedGpt2Small records execution state for the gpt2 small component.
type GeneratedGpt2Small struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGpt2Small() *GeneratedGpt2Small {
	return &GeneratedGpt2Small{Name: "gpt2_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGpt2Small) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGpt2Small) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
