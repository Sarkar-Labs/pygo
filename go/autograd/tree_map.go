package autograd

import "time"

// GeneratedTreeMap records execution state for the tree map component.
type GeneratedTreeMap struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTreeMap() *GeneratedTreeMap {
	return &GeneratedTreeMap{Name: "tree_map", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTreeMap) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTreeMap) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
