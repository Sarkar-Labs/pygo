package models

import "time"

// GeneratedResnetDistilled records execution state for the resnet distilled component.
type GeneratedResnetDistilled struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResnetDistilled() *GeneratedResnetDistilled {
	return &GeneratedResnetDistilled{Name: "resnet_distilled", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResnetDistilled) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResnetDistilled) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
