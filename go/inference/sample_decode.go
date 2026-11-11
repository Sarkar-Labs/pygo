package inference

import "time"

// GeneratedSampleDecode records execution state for the sample decode component.
type GeneratedSampleDecode struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSampleDecode() *GeneratedSampleDecode {
	return &GeneratedSampleDecode{Name: "sample_decode", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSampleDecode) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSampleDecode) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
