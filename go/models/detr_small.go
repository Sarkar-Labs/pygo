package models

import "time"

// GeneratedDetrSmall records execution state for the detr small component.
type GeneratedDetrSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrSmall() *GeneratedDetrSmall {
	return &GeneratedDetrSmall{Name: "detr_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
