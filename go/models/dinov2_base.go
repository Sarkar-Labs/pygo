package models

import "time"

// GeneratedDinov2Base records execution state for the dinov2 base component.
type GeneratedDinov2Base struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinov2Base() *GeneratedDinov2Base {
	return &GeneratedDinov2Base{Name: "dinov2_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinov2Base) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinov2Base) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
