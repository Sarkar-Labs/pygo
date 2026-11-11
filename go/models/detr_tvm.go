package models

import "time"

// GeneratedDetrTvm records execution state for the detr tvm component.
type GeneratedDetrTvm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrTvm() *GeneratedDetrTvm {
	return &GeneratedDetrTvm{Name: "detr_tvm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrTvm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrTvm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
