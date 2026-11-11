package models

import "time"

// GeneratedSamTvm records execution state for the sam tvm component.
type GeneratedSamTvm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamTvm() *GeneratedSamTvm {
	return &GeneratedSamTvm{Name: "sam_tvm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamTvm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamTvm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
