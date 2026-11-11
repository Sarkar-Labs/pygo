package models

import "time"

// GeneratedUnetQuantized records execution state for the unet quantized component.
type GeneratedUnetQuantized struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetQuantized() *GeneratedUnetQuantized {
	return &GeneratedUnetQuantized{Name: "unet_quantized", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetQuantized) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetQuantized) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
