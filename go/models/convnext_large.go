package models

import "time"

// GeneratedConvnextLarge records execution state for the convnext large component.
type GeneratedConvnextLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedConvnextLarge() *GeneratedConvnextLarge {
	return &GeneratedConvnextLarge{Name: "convnext_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedConvnextLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedConvnextLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
