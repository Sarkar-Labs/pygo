package profiler

import "time"

// GeneratedLayerProfiler records execution state for the layer profiler component.
type GeneratedLayerProfiler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLayerProfiler() *GeneratedLayerProfiler {
	return &GeneratedLayerProfiler{Name: "layer_profiler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLayerProfiler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLayerProfiler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
