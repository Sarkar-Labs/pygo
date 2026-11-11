package models

import "time"

// GeneratedDetrLarge records execution state for the detr large component.
type GeneratedDetrLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrLarge() *GeneratedDetrLarge {
	return &GeneratedDetrLarge{Name: "detr_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
