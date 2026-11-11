package models

import "time"

// GeneratedUnetXlarge records execution state for the unet xlarge component.
type GeneratedUnetXlarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetXlarge() *GeneratedUnetXlarge {
	return &GeneratedUnetXlarge{Name: "unet_xlarge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetXlarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetXlarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
