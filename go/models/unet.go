package models

import "time"

// GeneratedUnet records execution state for the unet component.
type GeneratedUnet struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnet() *GeneratedUnet { return &GeneratedUnet{Name: "unet", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnet) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnet) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
