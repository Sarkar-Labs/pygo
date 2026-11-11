package inference

import "time"

// GeneratedTopPSample records execution state for the top p sample component.
type GeneratedTopPSample struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTopPSample() *GeneratedTopPSample {
	return &GeneratedTopPSample{Name: "top_p_sample", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTopPSample) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTopPSample) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
