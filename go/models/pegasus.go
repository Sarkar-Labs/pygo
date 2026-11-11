package models

import "time"

// GeneratedPegasus records execution state for the pegasus component.
type GeneratedPegasus struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedPegasus() *GeneratedPegasus {
	return &GeneratedPegasus{Name: "pegasus", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedPegasus) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedPegasus) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
