package callbacks

import "time"

// GeneratedOnForward records execution state for the on forward component.
type GeneratedOnForward struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedOnForward() *GeneratedOnForward {
	return &GeneratedOnForward{Name: "on_forward", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedOnForward) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedOnForward) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
