package profiler

import "time"

// GeneratedGpuProfiler records execution state for the gpu profiler component.
type GeneratedGpuProfiler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGpuProfiler() *GeneratedGpuProfiler {
	return &GeneratedGpuProfiler{Name: "gpu_profiler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGpuProfiler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGpuProfiler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
