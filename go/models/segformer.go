package models

import "time"

// GeneratedSegformer records execution state for the segformer component.
type GeneratedSegformer struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSegformer() *GeneratedSegformer {
	return &GeneratedSegformer{Name: "segformer", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSegformer) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSegformer) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
