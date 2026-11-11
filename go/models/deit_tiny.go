package models

import "time"

// GeneratedDeitTiny records execution state for the deit tiny component.
type GeneratedDeitTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDeitTiny() *GeneratedDeitTiny {
	return &GeneratedDeitTiny{Name: "deit_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDeitTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDeitTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
