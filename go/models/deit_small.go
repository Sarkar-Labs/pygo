package models

import "time"

// GeneratedDeitSmall records execution state for the deit small component.
type GeneratedDeitSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDeitSmall() *GeneratedDeitSmall {
	return &GeneratedDeitSmall{Name: "deit_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDeitSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDeitSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
