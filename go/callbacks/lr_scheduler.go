package callbacks

import "time"

// GeneratedLrScheduler records execution state for the lr scheduler component.
type GeneratedLrScheduler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLrScheduler() *GeneratedLrScheduler {
	return &GeneratedLrScheduler{Name: "lr_scheduler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLrScheduler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLrScheduler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
