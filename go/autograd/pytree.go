package autograd

import "time"

// GeneratedPytree records execution state for the pytree component.
type GeneratedPytree struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedPytree() *GeneratedPytree {
	return &GeneratedPytree{Name: "pytree", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedPytree) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedPytree) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
