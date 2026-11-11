package models

import "time"

// GeneratedVitPretrained records execution state for the vit pretrained component.
type GeneratedVitPretrained struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitPretrained() *GeneratedVitPretrained {
	return &GeneratedVitPretrained{Name: "vit_pretrained", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitPretrained) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitPretrained) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
