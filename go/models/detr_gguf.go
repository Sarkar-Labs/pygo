package models

import "time"

// GeneratedDetrGguf records execution state for the detr gguf component.
type GeneratedDetrGguf struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrGguf() *GeneratedDetrGguf {
	return &GeneratedDetrGguf{Name: "detr_gguf", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrGguf) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrGguf) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
