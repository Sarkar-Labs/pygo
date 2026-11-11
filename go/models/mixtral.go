package models

import "time"

// GeneratedMixtral records execution state for the mixtral component.
type GeneratedMixtral struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMixtral() *GeneratedMixtral {
	return &GeneratedMixtral{Name: "mixtral", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMixtral) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMixtral) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
