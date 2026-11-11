package models

import "time"

// GeneratedGptV4 records execution state for the gpt v4 component.
type GeneratedGptV4 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptV4() *GeneratedGptV4 {
	return &GeneratedGptV4{Name: "gpt_v4", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptV4) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptV4) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
