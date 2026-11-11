package profiler

import "time"

// GeneratedMemoryProfiler records execution state for the memory profiler component.
type GeneratedMemoryProfiler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMemoryProfiler() *GeneratedMemoryProfiler {
	return &GeneratedMemoryProfiler{Name: "memory_profiler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMemoryProfiler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMemoryProfiler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
