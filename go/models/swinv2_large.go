package models

import "time"

// GeneratedSwinv2Large records execution state for the swinv2 large component.
type GeneratedSwinv2Large struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSwinv2Large() *GeneratedSwinv2Large {
	return &GeneratedSwinv2Large{Name: "swinv2_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSwinv2Large) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSwinv2Large) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
