package autograd

import "time"

// GeneratedJacobian records execution state for the jacobian component.
type GeneratedJacobian struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedJacobian() *GeneratedJacobian {
	return &GeneratedJacobian{Name: "jacobian", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedJacobian) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedJacobian) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
