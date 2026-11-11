package autograd

import "time"

// GeneratedZero records execution state for the zero component.
type GeneratedZero struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedZero() *GeneratedZero { return &GeneratedZero{Name: "zero", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedZero) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedZero) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
