package data

import "time"

// GeneratedVideoDataset records execution state for the video dataset component.
type GeneratedVideoDataset struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVideoDataset() *GeneratedVideoDataset {
	return &GeneratedVideoDataset{Name: "video_dataset", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVideoDataset) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVideoDataset) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
