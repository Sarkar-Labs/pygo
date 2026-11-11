package models

import "time"

// GeneratedConvnextTiny records execution state for the convnext tiny component.
type GeneratedConvnextTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedConvnextTiny() *GeneratedConvnextTiny {
	return &GeneratedConvnextTiny{Name: "convnext_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedConvnextTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedConvnextTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
