package data

import "time"

// GeneratedStreamDataset records execution state for the stream dataset component.
type GeneratedStreamDataset struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedStreamDataset() *GeneratedStreamDataset {
	return &GeneratedStreamDataset{Name: "stream_dataset", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedStreamDataset) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedStreamDataset) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
