package regularizers

import "time"

// GeneratedDropconnect records execution state for the dropconnect component.
type GeneratedDropconnect struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDropconnect() *GeneratedDropconnect {
	return &GeneratedDropconnect{Name: "dropconnect", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDropconnect) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDropconnect) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
