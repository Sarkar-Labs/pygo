package profiler

import "time"

// GeneratedModelProfiler records execution state for the model profiler component.
type GeneratedModelProfiler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedModelProfiler() *GeneratedModelProfiler {
	return &GeneratedModelProfiler{Name: "model_profiler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedModelProfiler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedModelProfiler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
