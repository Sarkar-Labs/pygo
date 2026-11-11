package profiler

import "time"

// GeneratedOpProfiler records execution state for the op profiler component.
type GeneratedOpProfiler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedOpProfiler() *GeneratedOpProfiler {
	return &GeneratedOpProfiler{Name: "op_profiler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedOpProfiler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedOpProfiler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
