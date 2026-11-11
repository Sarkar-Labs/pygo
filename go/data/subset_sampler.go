package data

import "time"

// GeneratedSubsetSampler records execution state for the subset sampler component.
type GeneratedSubsetSampler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSubsetSampler() *GeneratedSubsetSampler {
	return &GeneratedSubsetSampler{Name: "subset_sampler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSubsetSampler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSubsetSampler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
