package transforms

import "time"

// GeneratedRandomFlip records execution state for the random flip component.
type GeneratedRandomFlip struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRandomFlip() *GeneratedRandomFlip {
	return &GeneratedRandomFlip{Name: "random_flip", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRandomFlip) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRandomFlip) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
