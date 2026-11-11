package autograd

import "time"

// GeneratedGradFn records execution state for the grad fn component.
type GeneratedGradFn struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGradFn() *GeneratedGradFn {
	return &GeneratedGradFn{Name: "grad_fn", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGradFn) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGradFn) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
