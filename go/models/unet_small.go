package models

import "time"

// GeneratedUnetSmall records execution state for the unet small component.
type GeneratedUnetSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetSmall() *GeneratedUnetSmall {
	return &GeneratedUnetSmall{Name: "unet_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
