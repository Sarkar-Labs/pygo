package models

import "time"

// GeneratedUnetPretrained records execution state for the unet pretrained component.
type GeneratedUnetPretrained struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetPretrained() *GeneratedUnetPretrained {
	return &GeneratedUnetPretrained{Name: "unet_pretrained", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetPretrained) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetPretrained) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
