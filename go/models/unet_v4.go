package models

import "time"

// GeneratedUnetV4 records execution state for the unet v4 component.
type GeneratedUnetV4 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetV4() *GeneratedUnetV4 {
	return &GeneratedUnetV4{Name: "unet_v4", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetV4) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetV4) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
