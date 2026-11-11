package models

import "time"

// GeneratedUnetBase records execution state for the unet base component.
type GeneratedUnetBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetBase() *GeneratedUnetBase {
	return &GeneratedUnetBase{Name: "unet_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
