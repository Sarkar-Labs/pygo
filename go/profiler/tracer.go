package profiler

import "time"

// GeneratedTracer records execution state for the tracer component.
type GeneratedTracer struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTracer() *GeneratedTracer {
	return &GeneratedTracer{Name: "tracer", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTracer) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTracer) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
