package models

import "time"

// GeneratedUnetV2 records execution state for the unet v2 component.
type GeneratedUnetV2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetV2() *GeneratedUnetV2 {
	return &GeneratedUnetV2{Name: "unet_v2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetV2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetV2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
