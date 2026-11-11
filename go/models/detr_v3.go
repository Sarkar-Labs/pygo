package models

import "time"

// GeneratedDetrV3 records execution state for the detr v3 component.
type GeneratedDetrV3 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrV3() *GeneratedDetrV3 {
	return &GeneratedDetrV3{Name: "detr_v3", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrV3) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrV3) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
