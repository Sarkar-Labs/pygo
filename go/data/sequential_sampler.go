package data

import "time"

// GeneratedSequentialSampler records execution state for the sequential sampler component.
type GeneratedSequentialSampler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSequentialSampler() *GeneratedSequentialSampler {
	return &GeneratedSequentialSampler{Name: "sequential_sampler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSequentialSampler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSequentialSampler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
