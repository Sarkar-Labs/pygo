package autograd

import "time"

// GeneratedJvp records execution state for the jvp component.
type GeneratedJvp struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedJvp() *GeneratedJvp { return &GeneratedJvp{Name: "jvp", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedJvp) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedJvp) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
