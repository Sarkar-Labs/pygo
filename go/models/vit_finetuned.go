package models

import "time"

// GeneratedVitFinetuned records execution state for the vit finetuned component.
type GeneratedVitFinetuned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitFinetuned() *GeneratedVitFinetuned {
	return &GeneratedVitFinetuned{Name: "vit_finetuned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitFinetuned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitFinetuned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
