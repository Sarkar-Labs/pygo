package logging

import "time"

// GeneratedLevel records execution state for the level component.
type GeneratedLevel struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLevel() *GeneratedLevel {
	return &GeneratedLevel{Name: "level", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLevel) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLevel) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
