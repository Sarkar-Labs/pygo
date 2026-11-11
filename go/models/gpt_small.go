package models

import "time"

// GeneratedGptSmall records execution state for the gpt small component.
type GeneratedGptSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptSmall() *GeneratedGptSmall {
	return &GeneratedGptSmall{Name: "gpt_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
