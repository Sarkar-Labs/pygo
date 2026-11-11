package models

import "time"

// GeneratedGptLarge records execution state for the gpt large component.
type GeneratedGptLarge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptLarge() *GeneratedGptLarge {
	return &GeneratedGptLarge{Name: "gpt_large", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptLarge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptLarge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
