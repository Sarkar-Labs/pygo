package transforms

import "time"

// GeneratedMosaic records execution state for the mosaic component.
type GeneratedMosaic struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMosaic() *GeneratedMosaic {
	return &GeneratedMosaic{Name: "mosaic", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMosaic) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMosaic) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
