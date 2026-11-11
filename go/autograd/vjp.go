package autograd

import "time"

// GeneratedVjp records execution state for the vjp component.
type GeneratedVjp struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVjp() *GeneratedVjp { return &GeneratedVjp{Name: "vjp", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVjp) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVjp) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
