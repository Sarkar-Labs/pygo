package data

import "time"

// GeneratedSuperglue records execution state for the superglue component.
type GeneratedSuperglue struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSuperglue() *GeneratedSuperglue {
	return &GeneratedSuperglue{Name: "superglue", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSuperglue) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSuperglue) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
