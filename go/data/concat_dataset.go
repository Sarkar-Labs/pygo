package data

import "time"

// GeneratedConcatDataset records execution state for the concat dataset component.
type GeneratedConcatDataset struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedConcatDataset() *GeneratedConcatDataset {
	return &GeneratedConcatDataset{Name: "concat_dataset", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedConcatDataset) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedConcatDataset) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
