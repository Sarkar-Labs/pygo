package callbacks

import "time"

// GeneratedWeightMonitor records execution state for the weight monitor component.
type GeneratedWeightMonitor struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWeightMonitor() *GeneratedWeightMonitor {
	return &GeneratedWeightMonitor{Name: "weight_monitor", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWeightMonitor) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWeightMonitor) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
