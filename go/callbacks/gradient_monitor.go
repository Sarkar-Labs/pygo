package callbacks

import "time"

// GeneratedGradientMonitor records execution state for the gradient monitor component.
type GeneratedGradientMonitor struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGradientMonitor() *GeneratedGradientMonitor {
	return &GeneratedGradientMonitor{Name: "gradient_monitor", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGradientMonitor) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGradientMonitor) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
