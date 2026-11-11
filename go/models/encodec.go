package models

import "time"

// GeneratedEncodec records execution state for the encodec component.
type GeneratedEncodec struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEncodec() *GeneratedEncodec {
	return &GeneratedEncodec{Name: "encodec", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEncodec) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEncodec) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
