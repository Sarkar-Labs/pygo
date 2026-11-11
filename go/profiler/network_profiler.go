package profiler

import "time"

// GeneratedNetworkProfiler records execution state for the network profiler component.
type GeneratedNetworkProfiler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedNetworkProfiler() *GeneratedNetworkProfiler {
	return &GeneratedNetworkProfiler{Name: "network_profiler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedNetworkProfiler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedNetworkProfiler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
