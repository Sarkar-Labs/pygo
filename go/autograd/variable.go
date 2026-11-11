package autograd

import "time"

// GeneratedVariable records execution state for the variable component.
type GeneratedVariable struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVariable() *GeneratedVariable {
	return &GeneratedVariable{Name: "variable", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVariable) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVariable) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
