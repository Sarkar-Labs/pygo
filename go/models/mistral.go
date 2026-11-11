package models

import "time"

// GeneratedMistral records execution state for the mistral component.
type GeneratedMistral struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMistral() *GeneratedMistral {
	return &GeneratedMistral{Name: "mistral", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMistral) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMistral) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
