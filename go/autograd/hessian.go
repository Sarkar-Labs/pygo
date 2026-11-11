package autograd

import "time"

// GeneratedHessian records execution state for the hessian component.
type GeneratedHessian struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedHessian() *GeneratedHessian {
	return &GeneratedHessian{Name: "hessian", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedHessian) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedHessian) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
