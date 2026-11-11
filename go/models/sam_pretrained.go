package models

import "time"

// GeneratedSamPretrained records execution state for the sam pretrained component.
type GeneratedSamPretrained struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamPretrained() *GeneratedSamPretrained {
	return &GeneratedSamPretrained{Name: "sam_pretrained", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamPretrained) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamPretrained) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
