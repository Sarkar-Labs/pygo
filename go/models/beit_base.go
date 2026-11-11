package models

import "time"

// GeneratedBeitBase records execution state for the beit base component.
type GeneratedBeitBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBeitBase() *GeneratedBeitBase {
	return &GeneratedBeitBase{Name: "beit_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBeitBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBeitBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
