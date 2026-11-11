package models

import "time"

// GeneratedBeitLarge records execution state for the beit large component.
type GeneratedBeitLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBeitLarge() *GeneratedBeitLarge {
	return &GeneratedBeitLarge{Name: "beit_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBeitLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBeitLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
