package models

import "time"

// GeneratedSamBase records execution state for the sam base component.
type GeneratedSamBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamBase() *GeneratedSamBase {
	return &GeneratedSamBase{Name: "sam_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
