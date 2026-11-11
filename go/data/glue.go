package data

import "time"

// GeneratedGlue records execution state for the glue component.
type GeneratedGlue struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGlue() *GeneratedGlue { return &GeneratedGlue{Name: "glue", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGlue) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGlue) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
