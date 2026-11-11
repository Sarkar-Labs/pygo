package models

import "time"

// GeneratedGptV1 records execution state for the gpt v1 component.
type GeneratedGptV1 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptV1() *GeneratedGptV1 {
	return &GeneratedGptV1{Name: "gpt_v1", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptV1) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptV1) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
