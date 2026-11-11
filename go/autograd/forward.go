package autograd

import "time"

// GeneratedForward records execution state for the forward component.
type GeneratedForward struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedForward() *GeneratedForward {
	return &GeneratedForward{Name: "forward", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedForward) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedForward) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
