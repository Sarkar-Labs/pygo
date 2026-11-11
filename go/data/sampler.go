package data

import "time"

// GeneratedSampler records execution state for the sampler component.
type GeneratedSampler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSampler() *GeneratedSampler {
	return &GeneratedSampler{Name: "sampler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSampler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSampler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
