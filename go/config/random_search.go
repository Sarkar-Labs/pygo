package config

import "time"

// GeneratedRandomSearch records execution state for the random search component.
type GeneratedRandomSearch struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRandomSearch() *GeneratedRandomSearch {
	return &GeneratedRandomSearch{Name: "random_search", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRandomSearch) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRandomSearch) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
