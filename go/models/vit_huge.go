package models

import "time"

// GeneratedVitHuge records execution state for the vit huge component.
type GeneratedVitHuge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVitHuge() *GeneratedVitHuge {
	return &GeneratedVitHuge{Name: "vit_huge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVitHuge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVitHuge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
