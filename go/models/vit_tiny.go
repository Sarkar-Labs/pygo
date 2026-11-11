package models

import "time"

// GeneratedVitTiny records execution state for the vit tiny component.
type GeneratedVitTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitTiny() *GeneratedVitTiny {
	return &GeneratedVitTiny{Name: "vit_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
