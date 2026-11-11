package models

import "time"

// GeneratedGpt2Medium records execution state for the gpt2 medium component.
type GeneratedGpt2Medium struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGpt2Medium() *GeneratedGpt2Medium {
	return &GeneratedGpt2Medium{Name: "gpt2_medium", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGpt2Medium) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGpt2Medium) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
