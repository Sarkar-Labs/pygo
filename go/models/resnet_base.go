package models

import "time"

// GeneratedResnetBase records execution state for the resnet base component.
type GeneratedResnetBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetBase() *GeneratedResnetBase {
	return &GeneratedResnetBase{Name: "resnet_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
