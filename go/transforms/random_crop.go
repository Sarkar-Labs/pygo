package transforms

import "time"

// GeneratedRandomCrop records execution state for the random crop component.
type GeneratedRandomCrop struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRandomCrop() *GeneratedRandomCrop {
	return &GeneratedRandomCrop{Name: "random_crop", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRandomCrop) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRandomCrop) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
