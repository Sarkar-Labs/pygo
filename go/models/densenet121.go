package models

import "time"

// GeneratedDensenet121 records execution state for the densenet121 component.
type GeneratedDensenet121 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDensenet121() *GeneratedDensenet121 {
	return &GeneratedDensenet121{Name: "densenet121", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDensenet121) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDensenet121) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
