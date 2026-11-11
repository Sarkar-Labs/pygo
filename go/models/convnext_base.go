package models

import "time"

// GeneratedConvnextBase records execution state for the convnext base component.
type GeneratedConvnextBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedConvnextBase() *GeneratedConvnextBase {
	return &GeneratedConvnextBase{Name: "convnext_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedConvnextBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedConvnextBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
