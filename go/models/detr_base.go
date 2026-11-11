package models

import "time"

// GeneratedDetrBase records execution state for the detr base component.
type GeneratedDetrBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrBase() *GeneratedDetrBase {
	return &GeneratedDetrBase{Name: "detr_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
