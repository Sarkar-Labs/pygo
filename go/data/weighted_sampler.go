package data

import "time"

// GeneratedWeightedSampler records execution state for the weighted sampler component.
type GeneratedWeightedSampler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWeightedSampler() *GeneratedWeightedSampler {
	return &GeneratedWeightedSampler{Name: "weighted_sampler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWeightedSampler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWeightedSampler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
