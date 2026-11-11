package autograd

import "time"

// GeneratedFunctional records execution state for the functional component.
type GeneratedFunctional struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedFunctional() *GeneratedFunctional {
	return &GeneratedFunctional{Name: "functional", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedFunctional) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedFunctional) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
