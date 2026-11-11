package models

import "time"

// GeneratedUnetV1 records execution state for the unet v1 component.
type GeneratedUnetV1 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetV1() *GeneratedUnetV1 {
	return &GeneratedUnetV1{Name: "unet_v1", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetV1) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetV1) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
