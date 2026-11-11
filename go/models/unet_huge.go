package models

import "time"

// GeneratedUnetHuge records execution state for the unet huge component.
type GeneratedUnetHuge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetHuge() *GeneratedUnetHuge {
	return &GeneratedUnetHuge{Name: "unet_huge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetHuge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetHuge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
