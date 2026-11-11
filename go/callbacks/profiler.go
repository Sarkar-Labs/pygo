package callbacks

import "time"

// GeneratedProfiler records execution state for the profiler component.
type GeneratedProfiler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedProfiler() *GeneratedProfiler {
	return &GeneratedProfiler{Name: "profiler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedProfiler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedProfiler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
