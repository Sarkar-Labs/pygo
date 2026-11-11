package transforms

import "time"

// GeneratedFiveCrop records execution state for the five crop component.
type GeneratedFiveCrop struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedFiveCrop() *GeneratedFiveCrop {
	return &GeneratedFiveCrop{Name: "five_crop", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedFiveCrop) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedFiveCrop) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
