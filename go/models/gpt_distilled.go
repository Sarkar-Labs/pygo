package models

import "time"

// GeneratedGptDistilled records execution state for the gpt distilled component.
type GeneratedGptDistilled struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptDistilled() *GeneratedGptDistilled {
	return &GeneratedGptDistilled{Name: "gpt_distilled", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptDistilled) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptDistilled) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
