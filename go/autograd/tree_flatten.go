package autograd

import "time"

// GeneratedTreeFlatten records execution state for the tree flatten component.
type GeneratedTreeFlatten struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTreeFlatten() *GeneratedTreeFlatten {
	return &GeneratedTreeFlatten{Name: "tree_flatten", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTreeFlatten) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTreeFlatten) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
