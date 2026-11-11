package profiler

import "time"

// GeneratedCpuProfiler records execution state for the cpu profiler component.
type GeneratedCpuProfiler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCpuProfiler() *GeneratedCpuProfiler {
	return &GeneratedCpuProfiler{Name: "cpu_profiler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCpuProfiler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCpuProfiler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
