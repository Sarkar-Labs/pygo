package callbacks

import "time"

// GeneratedEventSystem records execution state for the event system component.
type GeneratedEventSystem struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEventSystem() *GeneratedEventSystem {
	return &GeneratedEventSystem{Name: "event_system", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEventSystem) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEventSystem) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
