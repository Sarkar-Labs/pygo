package config

import "time"

// GeneratedSearchSpace records execution state for the search space component.
type GeneratedSearchSpace struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSearchSpace() *GeneratedSearchSpace {
	return &GeneratedSearchSpace{Name: "search_space", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSearchSpace) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSearchSpace) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
