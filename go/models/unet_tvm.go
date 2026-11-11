package models

import "time"

// GeneratedUnetTvm records execution state for the unet tvm component.
type GeneratedUnetTvm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetTvm() *GeneratedUnetTvm {
	return &GeneratedUnetTvm{Name: "unet_tvm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetTvm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetTvm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
