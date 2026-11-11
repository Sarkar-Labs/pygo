package data

import "time"

// GeneratedDistributedSampler records execution state for the distributed sampler component.
type GeneratedDistributedSampler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDistributedSampler() *GeneratedDistributedSampler {
	return &GeneratedDistributedSampler{Name: "distributed_sampler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDistributedSampler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDistributedSampler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
