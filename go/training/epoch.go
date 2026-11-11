package training

import "time"

// GeneratedEpoch records execution state for the epoch component.
type GeneratedEpoch struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEpoch() *GeneratedEpoch {
	return &GeneratedEpoch{Name: "epoch", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEpoch) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEpoch) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
