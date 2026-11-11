package data

import "time"

// GeneratedRandomSampler records execution state for the random sampler component.
type GeneratedRandomSampler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRandomSampler() *GeneratedRandomSampler {
	return &GeneratedRandomSampler{Name: "random_sampler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRandomSampler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRandomSampler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
