package models

import "time"

// GeneratedSamV4 records execution state for the sam v4 component.
type GeneratedSamV4 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamV4() *GeneratedSamV4 {
	return &GeneratedSamV4{Name: "sam_v4", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamV4) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamV4) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
