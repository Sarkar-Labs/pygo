package transforms

import "time"

// GeneratedRandomAffine records execution state for the random affine component.
type GeneratedRandomAffine struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRandomAffine() *GeneratedRandomAffine {
	return &GeneratedRandomAffine{Name: "random_affine", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRandomAffine) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRandomAffine) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
