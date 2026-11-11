package profiler

import "time"

// GeneratedBandwidthProfiler records execution state for the bandwidth profiler component.
type GeneratedBandwidthProfiler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBandwidthProfiler() *GeneratedBandwidthProfiler {
	return &GeneratedBandwidthProfiler{Name: "bandwidth_profiler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBandwidthProfiler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBandwidthProfiler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
