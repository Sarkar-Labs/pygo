package callbacks

import "time"

// GeneratedTimer records execution state for the timer component.
type GeneratedTimer struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTimer() *GeneratedTimer {
	return &GeneratedTimer{Name: "timer", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTimer) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTimer) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
