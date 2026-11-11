package transforms

import "time"

// GeneratedRandomGrayscale records execution state for the random grayscale component.
type GeneratedRandomGrayscale struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRandomGrayscale() *GeneratedRandomGrayscale {
	return &GeneratedRandomGrayscale{Name: "random_grayscale", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRandomGrayscale) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRandomGrayscale) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
