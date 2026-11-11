package models

import "time"

// GeneratedVitDistilled records execution state for the vit distilled component.
type GeneratedVitDistilled struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitDistilled() *GeneratedVitDistilled {
	return &GeneratedVitDistilled{Name: "vit_distilled", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitDistilled) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitDistilled) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
