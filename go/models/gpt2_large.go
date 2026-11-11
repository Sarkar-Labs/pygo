package models

import "time"

// GeneratedGpt2Large records execution state for the gpt2 large component.
type GeneratedGpt2Large struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGpt2Large() *GeneratedGpt2Large {
	return &GeneratedGpt2Large{Name: "gpt2_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGpt2Large) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGpt2Large) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
