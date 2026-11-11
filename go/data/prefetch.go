package data

import "time"

// GeneratedPrefetch records execution state for the prefetch component.
type GeneratedPrefetch struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedPrefetch() *GeneratedPrefetch {
	return &GeneratedPrefetch{Name: "prefetch", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedPrefetch) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedPrefetch) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
