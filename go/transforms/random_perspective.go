package transforms

import "time"

// GeneratedRandomPerspective records execution state for the random perspective component.
type GeneratedRandomPerspective struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRandomPerspective() *GeneratedRandomPerspective {
	return &GeneratedRandomPerspective{Name: "random_perspective", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRandomPerspective) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRandomPerspective) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
