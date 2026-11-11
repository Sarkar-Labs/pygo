package transforms

import "time"

// GeneratedRandAugment records execution state for the rand augment component.
type GeneratedRandAugment struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRandAugment() *GeneratedRandAugment {
	return &GeneratedRandAugment{Name: "rand_augment", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRandAugment) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRandAugment) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
