package models

import "time"

// GeneratedVitTvm records execution state for the vit tvm component.
type GeneratedVitTvm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitTvm() *GeneratedVitTvm {
	return &GeneratedVitTvm{Name: "vit_tvm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitTvm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitTvm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
