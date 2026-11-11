package models

import "time"

// GeneratedPhi records execution state for the phi component.
type GeneratedPhi struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedPhi() *GeneratedPhi { return &GeneratedPhi{Name: "phi", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedPhi) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedPhi) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
