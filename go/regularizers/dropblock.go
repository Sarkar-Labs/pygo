package regularizers

import "time"

// GeneratedDropblock records execution state for the dropblock component.
type GeneratedDropblock struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDropblock() *GeneratedDropblock {
	return &GeneratedDropblock{Name: "dropblock", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDropblock) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDropblock) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
