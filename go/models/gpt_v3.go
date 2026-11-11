package models

import "time"

// GeneratedGptV3 records execution state for the gpt v3 component.
type GeneratedGptV3 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptV3() *GeneratedGptV3 {
	return &GeneratedGptV3{Name: "gpt_v3", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptV3) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptV3) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
