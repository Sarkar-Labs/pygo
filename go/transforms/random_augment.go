package transforms

import "time"

// GeneratedRandomAugment records execution state for the random augment component.
type GeneratedRandomAugment struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRandomAugment() *GeneratedRandomAugment {
	return &GeneratedRandomAugment{Name: "random_augment", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRandomAugment) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRandomAugment) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
