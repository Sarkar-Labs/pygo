package data

import "time"

// GeneratedIterableDataset records execution state for the iterable dataset component.
type GeneratedIterableDataset struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedIterableDataset() *GeneratedIterableDataset {
	return &GeneratedIterableDataset{Name: "iterable_dataset", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedIterableDataset) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedIterableDataset) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
