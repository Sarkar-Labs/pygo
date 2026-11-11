package training

import "time"

// GeneratedUpdate records execution state for the update component.
type GeneratedUpdate struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedUpdate() *GeneratedUpdate {
	return &GeneratedUpdate{Name: "update", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedUpdate) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedUpdate) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
