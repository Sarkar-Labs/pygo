package profiler

import "time"

// GeneratedKernelProfiler records execution state for the kernel profiler component.
type GeneratedKernelProfiler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedKernelProfiler() *GeneratedKernelProfiler {
	return &GeneratedKernelProfiler{Name: "kernel_profiler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedKernelProfiler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedKernelProfiler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
