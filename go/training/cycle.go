package training

import "time"

// GeneratedCycle records execution state for the cycle component.
type GeneratedCycle struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCycle() *GeneratedCycle {
	return &GeneratedCycle{Name: "cycle", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCycle) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCycle) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
