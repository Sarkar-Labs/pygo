package autograd

import "time"

// GeneratedTopologicalSort records execution state for the topological sort component.
type GeneratedTopologicalSort struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTopologicalSort() *GeneratedTopologicalSort {
	return &GeneratedTopologicalSort{Name: "topological_sort", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTopologicalSort) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTopologicalSort) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
