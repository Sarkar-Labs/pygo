package data

import "time"

// GeneratedSubsetDataset records execution state for the subset dataset component.
type GeneratedSubsetDataset struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSubsetDataset() *GeneratedSubsetDataset {
	return &GeneratedSubsetDataset{Name: "subset_dataset", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSubsetDataset) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSubsetDataset) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
