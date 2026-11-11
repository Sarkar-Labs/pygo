package models

import "time"

// GeneratedDetrPretrained records execution state for the detr pretrained component.
type GeneratedDetrPretrained struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrPretrained() *GeneratedDetrPretrained {
	return &GeneratedDetrPretrained{Name: "detr_pretrained", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrPretrained) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrPretrained) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
