package autograd

import "time"

// GeneratedDetach records execution state for the detach component.
type GeneratedDetach struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDetach() *GeneratedDetach {
	return &GeneratedDetach{Name: "detach", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDetach) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDetach) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
