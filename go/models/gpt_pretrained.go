package models

import "time"

// GeneratedGptPretrained records execution state for the gpt pretrained component.
type GeneratedGptPretrained struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptPretrained() *GeneratedGptPretrained {
	return &GeneratedGptPretrained{Name: "gpt_pretrained", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptPretrained) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptPretrained) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
