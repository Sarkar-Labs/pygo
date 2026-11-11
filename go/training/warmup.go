package training

import "time"

// GeneratedWarmup records execution state for the warmup component.
type GeneratedWarmup struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWarmup() *GeneratedWarmup {
	return &GeneratedWarmup{Name: "warmup", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWarmup) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWarmup) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
