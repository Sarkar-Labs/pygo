package models

import "time"

// GeneratedGptHuge records execution state for the gpt huge component.
type GeneratedGptHuge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptHuge() *GeneratedGptHuge {
	return &GeneratedGptHuge{Name: "gpt_huge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptHuge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptHuge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
