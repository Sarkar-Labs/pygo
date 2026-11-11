package inference

import "time"

// GeneratedTopKSample records execution state for the top k sample component.
type GeneratedTopKSample struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTopKSample() *GeneratedTopKSample {
	return &GeneratedTopKSample{Name: "top_k_sample", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTopKSample) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTopKSample) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
