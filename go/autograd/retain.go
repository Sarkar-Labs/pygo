package autograd

import "time"

// GeneratedRetain records execution state for the retain component.
type GeneratedRetain struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRetain() *GeneratedRetain {
	return &GeneratedRetain{Name: "retain", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRetain) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRetain) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
