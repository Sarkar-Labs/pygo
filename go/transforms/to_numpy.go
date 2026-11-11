package transforms

import "time"

// GeneratedToNumpy records execution state for the to numpy component.
type GeneratedToNumpy struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedToNumpy() *GeneratedToNumpy {
	return &GeneratedToNumpy{Name: "to_numpy", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedToNumpy) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedToNumpy) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
