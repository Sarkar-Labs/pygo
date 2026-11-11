package callbacks

import "time"

// GeneratedActivationMonitor records execution state for the activation monitor component.
type GeneratedActivationMonitor struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedActivationMonitor() *GeneratedActivationMonitor {
	return &GeneratedActivationMonitor{Name: "activation_monitor", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedActivationMonitor) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedActivationMonitor) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
