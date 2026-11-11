package models

import "time"

// GeneratedUnetLarge records execution state for the unet large component.
type GeneratedUnetLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetLarge() *GeneratedUnetLarge {
	return &GeneratedUnetLarge{Name: "unet_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
