package config

import "time"

// GeneratedGridSearch records execution state for the grid search component.
type GeneratedGridSearch struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGridSearch() *GeneratedGridSearch {
	return &GeneratedGridSearch{Name: "grid_search", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGridSearch) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGridSearch) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
