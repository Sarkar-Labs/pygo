package models

import "time"

// GeneratedVitSmall records execution state for the vit small component.
type GeneratedVitSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitSmall() *GeneratedVitSmall {
	return &GeneratedVitSmall{Name: "vit_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
