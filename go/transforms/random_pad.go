package transforms

import "time"

// GeneratedRandomPad records execution state for the random pad component.
type GeneratedRandomPad struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRandomPad() *GeneratedRandomPad {
	return &GeneratedRandomPad{Name: "random_pad", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRandomPad) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRandomPad) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
