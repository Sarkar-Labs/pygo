package models

import "time"

// GeneratedDetrFinetuned records execution state for the detr finetuned component.
type GeneratedDetrFinetuned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetrFinetuned() *GeneratedDetrFinetuned {
	return &GeneratedDetrFinetuned{Name: "detr_finetuned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetrFinetuned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetrFinetuned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
