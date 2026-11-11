package models

import "time"

// GeneratedGptJ records execution state for the gpt j component.
type GeneratedGptJ struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptJ() *GeneratedGptJ { return &GeneratedGptJ{Name: "gpt_j", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptJ) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptJ) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
