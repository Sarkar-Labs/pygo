package models

import "time"

// GeneratedMarian records execution state for the marian component.
type GeneratedMarian struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMarian() *GeneratedMarian {
	return &GeneratedMarian{Name: "marian", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMarian) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMarian) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
