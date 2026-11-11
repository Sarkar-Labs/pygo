package models

import "time"

// GeneratedGpt2Xl records execution state for the gpt2 xl component.
type GeneratedGpt2Xl struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGpt2Xl() *GeneratedGpt2Xl {
	return &GeneratedGpt2Xl{Name: "gpt2_xl", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGpt2Xl) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGpt2Xl) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
