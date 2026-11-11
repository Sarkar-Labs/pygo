package data

import "time"

// GeneratedDataloader records execution state for the dataloader component.
type GeneratedDataloader struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDataloader() *GeneratedDataloader {
	return &GeneratedDataloader{Name: "dataloader", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDataloader) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDataloader) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }

// DataLoader is the minimal iterator contract consumed by the training runtime.
type DataLoader interface {
	Iter() chan *Batch
}
