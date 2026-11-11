package models

import "time"

// GeneratedUnetPlusPlus records execution state for the unet plus plus component.
type GeneratedUnetPlusPlus struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetPlusPlus() *GeneratedUnetPlusPlus {
	return &GeneratedUnetPlusPlus{Name: "unet_plus_plus", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetPlusPlus) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetPlusPlus) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
