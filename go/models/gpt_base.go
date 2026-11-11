package models

import "time"

// GeneratedGptBase records execution state for the gpt base component.
type GeneratedGptBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptBase() *GeneratedGptBase {
	return &GeneratedGptBase{Name: "gpt_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
