package models

import "time"

// GeneratedGptV2 records execution state for the gpt v2 component.
type GeneratedGptV2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptV2() *GeneratedGptV2 {
	return &GeneratedGptV2{Name: "gpt_v2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptV2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptV2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
