package models

import "time"

// GeneratedSamV1 records execution state for the sam v1 component.
type GeneratedSamV1 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamV1() *GeneratedSamV1 {
	return &GeneratedSamV1{Name: "sam_v1", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamV1) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamV1) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
