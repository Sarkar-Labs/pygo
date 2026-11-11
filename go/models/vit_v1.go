package models

import "time"

// GeneratedVitV1 records execution state for the vit v1 component.
type GeneratedVitV1 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitV1() *GeneratedVitV1 {
	return &GeneratedVitV1{Name: "vit_v1", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitV1) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitV1) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
