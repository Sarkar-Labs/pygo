package models

import "time"

// GeneratedSwinv2Base records execution state for the swinv2 base component.
type GeneratedSwinv2Base struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSwinv2Base() *GeneratedSwinv2Base {
	return &GeneratedSwinv2Base{Name: "swinv2_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSwinv2Base) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSwinv2Base) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
