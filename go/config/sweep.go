package config

import "time"

// GeneratedSweep records execution state for the sweep component.
type GeneratedSweep struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSweep() *GeneratedSweep {
	return &GeneratedSweep{Name: "sweep", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSweep) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSweep) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
