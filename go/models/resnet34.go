package models

import "time"

// GeneratedResnet34 records execution state for the resnet34 component.
type GeneratedResnet34 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnet34() *GeneratedResnet34 {
	return &GeneratedResnet34{Name: "resnet34", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnet34) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnet34) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
