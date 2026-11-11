package models

import "time"

// GeneratedVitV2 records execution state for the vit v2 component.
type GeneratedVitV2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitV2() *GeneratedVitV2 {
	return &GeneratedVitV2{Name: "vit_v2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitV2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitV2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
