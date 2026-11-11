package models

import "time"

// GeneratedDetrTiny records execution state for the detr tiny component.
type GeneratedDetrTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrTiny() *GeneratedDetrTiny {
	return &GeneratedDetrTiny{Name: "detr_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
