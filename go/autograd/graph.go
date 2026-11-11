package autograd

import "time"

// GeneratedGraph records execution state for the graph component.
type GeneratedGraph struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGraph() *GeneratedGraph {
	return &GeneratedGraph{Name: "graph", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGraph) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGraph) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
