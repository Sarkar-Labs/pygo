package transforms

import "time"

// GeneratedMixup records execution state for the mixup component.
type GeneratedMixup struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMixup() *GeneratedMixup {
	return &GeneratedMixup{Name: "mixup", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMixup) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMixup) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
