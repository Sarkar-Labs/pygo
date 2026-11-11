package autograd

import "time"

// GeneratedNode records execution state for the node component.
type GeneratedNode struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedNode() *GeneratedNode { return &GeneratedNode{Name: "node", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedNode) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedNode) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
