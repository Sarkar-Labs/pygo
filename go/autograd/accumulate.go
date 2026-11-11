package autograd

import "time"

// GeneratedAccumulate records execution state for the accumulate component.
type GeneratedAccumulate struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedAccumulate() *GeneratedAccumulate {
	return &GeneratedAccumulate{Name: "accumulate", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedAccumulate) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedAccumulate) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
