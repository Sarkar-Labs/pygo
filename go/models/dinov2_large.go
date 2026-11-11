package models

import "time"

// GeneratedDinov2Large records execution state for the dinov2 large component.
type GeneratedDinov2Large struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinov2Large() *GeneratedDinov2Large {
	return &GeneratedDinov2Large{Name: "dinov2_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinov2Large) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinov2Large) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
