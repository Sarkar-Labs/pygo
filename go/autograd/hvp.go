package autograd

import "time"

// GeneratedHvp records execution state for the hvp component.
type GeneratedHvp struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedHvp() *GeneratedHvp { return &GeneratedHvp{Name: "hvp", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedHvp) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedHvp) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
