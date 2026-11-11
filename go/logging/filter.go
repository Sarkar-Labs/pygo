package logging

import "time"

// GeneratedFilter records execution state for the filter component.
type GeneratedFilter struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedFilter() *GeneratedFilter {
	return &GeneratedFilter{Name: "filter", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedFilter) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedFilter) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
