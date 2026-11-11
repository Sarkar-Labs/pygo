package profiler

import "time"

// GeneratedCompare records execution state for the compare component.
type GeneratedCompare struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCompare() *GeneratedCompare {
	return &GeneratedCompare{Name: "compare", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCompare) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCompare) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
