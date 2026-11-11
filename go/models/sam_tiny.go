package models

import "time"

// GeneratedSamTiny records execution state for the sam tiny component.
type GeneratedSamTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamTiny() *GeneratedSamTiny {
	return &GeneratedSamTiny{Name: "sam_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
