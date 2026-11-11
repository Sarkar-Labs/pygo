package callbacks

import "time"

// GeneratedOnEpochBegin records execution state for the on epoch begin component.
type GeneratedOnEpochBegin struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedOnEpochBegin() *GeneratedOnEpochBegin {
	return &GeneratedOnEpochBegin{Name: "on_epoch_begin", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedOnEpochBegin) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedOnEpochBegin) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
