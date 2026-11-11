package profiler

import "time"

// GeneratedIoProfiler records execution state for the io profiler component.
type GeneratedIoProfiler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedIoProfiler() *GeneratedIoProfiler {
	return &GeneratedIoProfiler{Name: "io_profiler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedIoProfiler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedIoProfiler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
