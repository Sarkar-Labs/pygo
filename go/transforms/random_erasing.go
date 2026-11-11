package transforms

import "time"

// GeneratedRandomErasing records execution state for the random erasing component.
type GeneratedRandomErasing struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRandomErasing() *GeneratedRandomErasing {
	return &GeneratedRandomErasing{Name: "random_erasing", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRandomErasing) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRandomErasing) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
