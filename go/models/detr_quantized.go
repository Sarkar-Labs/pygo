package models

import "time"

// GeneratedDetrQuantized records execution state for the detr quantized component.
type GeneratedDetrQuantized struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrQuantized() *GeneratedDetrQuantized {
	return &GeneratedDetrQuantized{Name: "detr_quantized", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrQuantized) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrQuantized) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
