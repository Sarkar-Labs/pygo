package models

import "time"

// GeneratedSamHuge records execution state for the sam huge component.
type GeneratedSamHuge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamHuge() *GeneratedSamHuge {
	return &GeneratedSamHuge{Name: "sam_huge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamHuge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamHuge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
