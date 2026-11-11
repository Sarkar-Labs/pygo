package transforms

import "time"

// GeneratedAugmix records execution state for the augmix component.
type GeneratedAugmix struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedAugmix() *GeneratedAugmix {
	return &GeneratedAugmix{Name: "augmix", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedAugmix) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedAugmix) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
