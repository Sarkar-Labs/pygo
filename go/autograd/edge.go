package autograd

import "time"

// GeneratedEdge records execution state for the edge component.
type GeneratedEdge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEdge() *GeneratedEdge { return &GeneratedEdge{Name: "edge", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEdge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEdge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
