package models

import "time"

// GeneratedVitBase records execution state for the vit base component.
type GeneratedVitBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitBase() *GeneratedVitBase {
	return &GeneratedVitBase{Name: "vit_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
