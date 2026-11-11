package transforms

import "time"

// GeneratedTrivialAugment records execution state for the trivial augment component.
type GeneratedTrivialAugment struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTrivialAugment() *GeneratedTrivialAugment {
	return &GeneratedTrivialAugment{Name: "trivial_augment", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTrivialAugment) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTrivialAugment) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
