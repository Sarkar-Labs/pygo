package models

import "time"

// GeneratedGptTvm records execution state for the gpt tvm component.
type GeneratedGptTvm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGptTvm() *GeneratedGptTvm {
	return &GeneratedGptTvm{Name: "gpt_tvm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGptTvm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGptTvm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
