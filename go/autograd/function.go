package autograd

import "time"

// GeneratedFunction records execution state for the function component.
type GeneratedFunction struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedFunction() *GeneratedFunction {
	return &GeneratedFunction{Name: "function", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedFunction) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedFunction) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
