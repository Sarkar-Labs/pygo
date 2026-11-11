package data

import "time"

// GeneratedMapDataset records execution state for the map dataset component.
type GeneratedMapDataset struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMapDataset() *GeneratedMapDataset {
	return &GeneratedMapDataset{Name: "map_dataset", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMapDataset) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMapDataset) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
