package transforms

import "time"

// GeneratedAutoAugment records execution state for the auto augment component.
type GeneratedAutoAugment struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedAutoAugment() *GeneratedAutoAugment {
	return &GeneratedAutoAugment{Name: "auto_augment", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedAutoAugment) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedAutoAugment) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
