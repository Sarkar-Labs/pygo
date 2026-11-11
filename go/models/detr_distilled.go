package models

import "time"

// GeneratedDetrDistilled records execution state for the detr distilled component.
type GeneratedDetrDistilled struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrDistilled() *GeneratedDetrDistilled {
	return &GeneratedDetrDistilled{Name: "detr_distilled", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrDistilled) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrDistilled) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
