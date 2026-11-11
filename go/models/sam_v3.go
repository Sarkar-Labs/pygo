package models

import "time"

// GeneratedSamV3 records execution state for the sam v3 component.
type GeneratedSamV3 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamV3() *GeneratedSamV3 {
	return &GeneratedSamV3{Name: "sam_v3", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamV3) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamV3) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
