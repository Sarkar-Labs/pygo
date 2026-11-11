package models

import "time"

// GeneratedUnetDistilled records execution state for the unet distilled component.
type GeneratedUnetDistilled struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUnetDistilled() *GeneratedUnetDistilled {
	return &GeneratedUnetDistilled{Name: "unet_distilled", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUnetDistilled) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUnetDistilled) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
