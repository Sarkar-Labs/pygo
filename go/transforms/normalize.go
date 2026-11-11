package transforms

import "time"

// GeneratedNormalize records execution state for the normalize component.
type GeneratedNormalize struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedNormalize() *GeneratedNormalize {
	return &GeneratedNormalize{Name: "normalize", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedNormalize) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedNormalize) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
