package models

import "time"

// GeneratedDeitBase records execution state for the deit base component.
type GeneratedDeitBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDeitBase() *GeneratedDeitBase {
	return &GeneratedDeitBase{Name: "deit_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDeitBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDeitBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
