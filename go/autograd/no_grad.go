package autograd

import "time"

// GeneratedNoGrad records execution state for the no grad component.
type GeneratedNoGrad struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedNoGrad() *GeneratedNoGrad {
	return &GeneratedNoGrad{Name: "no_grad", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedNoGrad) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedNoGrad) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
