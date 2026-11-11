package models

import "time"

// GeneratedBartBase records execution state for the bart base component.
type GeneratedBartBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBartBase() *GeneratedBartBase {
	return &GeneratedBartBase{Name: "bart_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBartBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBartBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
