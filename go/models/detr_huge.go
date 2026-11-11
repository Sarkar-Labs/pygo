package models

import "time"

// GeneratedDetrHuge records execution state for the detr huge component.
type GeneratedDetrHuge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrHuge() *GeneratedDetrHuge {
	return &GeneratedDetrHuge{Name: "detr_huge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrHuge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrHuge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
