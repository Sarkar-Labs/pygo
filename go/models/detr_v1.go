package models

import "time"

// GeneratedDetrV1 records execution state for the detr v1 component.
type GeneratedDetrV1 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrV1() *GeneratedDetrV1 {
	return &GeneratedDetrV1{Name: "detr_v1", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrV1) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrV1) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
