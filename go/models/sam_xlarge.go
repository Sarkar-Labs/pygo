package models

import "time"

// GeneratedSamXlarge records execution state for the sam xlarge component.
type GeneratedSamXlarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamXlarge() *GeneratedSamXlarge {
	return &GeneratedSamXlarge{Name: "sam_xlarge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamXlarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamXlarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
