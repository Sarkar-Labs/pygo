package deploy

import "time"

// GeneratedVertex records execution state for the vertex component.
type GeneratedVertex struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVertex() *GeneratedVertex {
	return &GeneratedVertex{Name: "vertex", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVertex) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVertex) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
