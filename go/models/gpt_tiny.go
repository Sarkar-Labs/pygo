package models

import "time"

// GeneratedGptTiny records execution state for the gpt tiny component.
type GeneratedGptTiny struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptTiny() *GeneratedGptTiny {
	return &GeneratedGptTiny{Name: "gpt_tiny", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptTiny) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptTiny) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
