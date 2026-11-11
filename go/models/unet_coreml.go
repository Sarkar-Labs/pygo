package models

import "time"

// GeneratedUnetCoreml records execution state for the unet coreml component.
type GeneratedUnetCoreml struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetCoreml() *GeneratedUnetCoreml {
	return &GeneratedUnetCoreml{Name: "unet_coreml", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetCoreml) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetCoreml) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
