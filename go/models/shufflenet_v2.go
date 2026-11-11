package models

import "time"

// GeneratedShufflenetV2 records execution state for the shufflenet v2 component.
type GeneratedShufflenetV2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedShufflenetV2() *GeneratedShufflenetV2 {
	return &GeneratedShufflenetV2{Name: "shufflenet_v2", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedShufflenetV2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedShufflenetV2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
