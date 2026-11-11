package autograd

import "time"

// GeneratedBackward records execution state for the backward component.
type GeneratedBackward struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBackward() *GeneratedBackward {
	return &GeneratedBackward{Name: "backward", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBackward) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBackward) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
