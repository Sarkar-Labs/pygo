package models

import "time"

// GeneratedShufflenetV1 records execution state for the shufflenet v1 component.
type GeneratedShufflenetV1 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedShufflenetV1() *GeneratedShufflenetV1 {
	return &GeneratedShufflenetV1{Name: "shufflenet_v1", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedShufflenetV1) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedShufflenetV1) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
