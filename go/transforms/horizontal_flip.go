package transforms

import "time"

// GeneratedHorizontalFlip records execution state for the horizontal flip component.
type GeneratedHorizontalFlip struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedHorizontalFlip() *GeneratedHorizontalFlip {
	return &GeneratedHorizontalFlip{Name: "horizontal_flip", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedHorizontalFlip) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedHorizontalFlip) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
