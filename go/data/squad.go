package data

import "time"

// GeneratedSquad records execution state for the squad component.
type GeneratedSquad struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSquad() *GeneratedSquad {
	return &GeneratedSquad{Name: "squad", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSquad) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSquad) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
