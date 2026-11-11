package models

import "time"

// GeneratedUnetTiny records execution state for the unet tiny component.
type GeneratedUnetTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetTiny() *GeneratedUnetTiny {
	return &GeneratedUnetTiny{Name: "unet_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
