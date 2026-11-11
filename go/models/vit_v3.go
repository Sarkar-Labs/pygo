package models

import "time"

// GeneratedVitV3 records execution state for the vit v3 component.
type GeneratedVitV3 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitV3() *GeneratedVitV3 {
	return &GeneratedVitV3{Name: "vit_v3", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitV3) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitV3) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
