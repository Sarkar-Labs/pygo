package training

import "time"

// GeneratedCosine records execution state for the cosine component.
type GeneratedCosine struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCosine() *GeneratedCosine {
	return &GeneratedCosine{Name: "cosine", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCosine) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCosine) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
