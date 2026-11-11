package models

import "time"

// GeneratedVitLarge records execution state for the vit large component.
type GeneratedVitLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitLarge() *GeneratedVitLarge {
	return &GeneratedVitLarge{Name: "vit_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
