package models

import "time"

// GeneratedFalcon records execution state for the falcon component.
type GeneratedFalcon struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedFalcon() *GeneratedFalcon {
	return &GeneratedFalcon{Name: "falcon", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedFalcon) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedFalcon) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
