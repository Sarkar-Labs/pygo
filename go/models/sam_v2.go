package models

import "time"

// GeneratedSamV2 records execution state for the sam v2 component.
type GeneratedSamV2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamV2() *GeneratedSamV2 {
	return &GeneratedSamV2{Name: "sam_v2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamV2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamV2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
