package models

import "time"

// GeneratedSamDistilled records execution state for the sam distilled component.
type GeneratedSamDistilled struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSamDistilled() *GeneratedSamDistilled {
	return &GeneratedSamDistilled{Name: "sam_distilled", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSamDistilled) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSamDistilled) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
