package callbacks

import "time"

// GeneratedOnEpochEnd records execution state for the on epoch end component.
type GeneratedOnEpochEnd struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedOnEpochEnd() *GeneratedOnEpochEnd {
	return &GeneratedOnEpochEnd{Name: "on_epoch_end", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedOnEpochEnd) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedOnEpochEnd) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
