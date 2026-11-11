package models

import "time"

// GeneratedDetrXlarge records execution state for the detr xlarge component.
type GeneratedDetrXlarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrXlarge() *GeneratedDetrXlarge {
	return &GeneratedDetrXlarge{Name: "detr_xlarge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrXlarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrXlarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
