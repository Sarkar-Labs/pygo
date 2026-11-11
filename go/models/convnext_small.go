package models

import "time"

// GeneratedConvnextSmall records execution state for the convnext small component.
type GeneratedConvnextSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedConvnextSmall() *GeneratedConvnextSmall {
	return &GeneratedConvnextSmall{Name: "convnext_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedConvnextSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedConvnextSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
