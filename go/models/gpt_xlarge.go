package models

import "time"

// GeneratedGptXlarge records execution state for the gpt xlarge component.
type GeneratedGptXlarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptXlarge() *GeneratedGptXlarge {
	return &GeneratedGptXlarge{Name: "gpt_xlarge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptXlarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptXlarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
