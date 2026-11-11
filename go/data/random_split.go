package data

import "time"

// GeneratedRandomSplit records execution state for the random split component.
type GeneratedRandomSplit struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRandomSplit() *GeneratedRandomSplit {
	return &GeneratedRandomSplit{Name: "random_split", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRandomSplit) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRandomSplit) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
