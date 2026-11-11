package data

import "time"

// GeneratedTextDataset records execution state for the text dataset component.
type GeneratedTextDataset struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTextDataset() *GeneratedTextDataset {
	return &GeneratedTextDataset{Name: "text_dataset", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTextDataset) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTextDataset) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
