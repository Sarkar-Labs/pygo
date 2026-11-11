package models

import "time"

// GeneratedDetrV2 records execution state for the detr v2 component.
type GeneratedDetrV2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrV2() *GeneratedDetrV2 {
	return &GeneratedDetrV2{Name: "detr_v2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrV2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrV2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
