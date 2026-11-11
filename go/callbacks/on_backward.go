package callbacks

import "time"

// GeneratedOnBackward records execution state for the on backward component.
type GeneratedOnBackward struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedOnBackward() *GeneratedOnBackward {
	return &GeneratedOnBackward{Name: "on_backward", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedOnBackward) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedOnBackward) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
