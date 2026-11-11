package models

import "time"

// GeneratedSamSmall records execution state for the sam small component.
type GeneratedSamSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamSmall() *GeneratedSamSmall {
	return &GeneratedSamSmall{Name: "sam_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
