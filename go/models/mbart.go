package models

import "time"

// GeneratedMbart records execution state for the mbart component.
type GeneratedMbart struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMbart() *GeneratedMbart {
	return &GeneratedMbart{Name: "mbart", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMbart) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMbart) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
