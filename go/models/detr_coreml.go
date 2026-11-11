package models

import "time"

// GeneratedDetrCoreml records execution state for the detr coreml component.
type GeneratedDetrCoreml struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrCoreml() *GeneratedDetrCoreml {
	return &GeneratedDetrCoreml{Name: "detr_coreml", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrCoreml) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrCoreml) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
