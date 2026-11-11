package models

import "time"

// GeneratedUnetV3 records execution state for the unet v3 component.
type GeneratedUnetV3 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetV3() *GeneratedUnetV3 {
	return &GeneratedUnetV3{Name: "unet_v3", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetV3) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetV3) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
