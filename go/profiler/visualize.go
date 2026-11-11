package profiler

import "time"

// GeneratedVisualize records execution state for the visualize component.
type GeneratedVisualize struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVisualize() *GeneratedVisualize {
	return &GeneratedVisualize{Name: "visualize", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVisualize) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVisualize) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
