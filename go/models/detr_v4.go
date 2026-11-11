package models

import "time"

// GeneratedDetrV4 records execution state for the detr v4 component.
type GeneratedDetrV4 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrV4() *GeneratedDetrV4 {
	return &GeneratedDetrV4{Name: "detr_v4", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrV4) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrV4) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
