package models

import "time"

// GeneratedVitV4 records execution state for the vit v4 component.
type GeneratedVitV4 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitV4() *GeneratedVitV4 {
	return &GeneratedVitV4{Name: "vit_v4", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitV4) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitV4) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
