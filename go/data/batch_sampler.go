package data

import "time"

// GeneratedBatchSampler records execution state for the batch sampler component.
type GeneratedBatchSampler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBatchSampler() *GeneratedBatchSampler {
	return &GeneratedBatchSampler{Name: "batch_sampler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBatchSampler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBatchSampler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
