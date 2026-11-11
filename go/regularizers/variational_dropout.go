package regularizers

import "time"

// GeneratedVariationalDropout records execution state for the variational dropout component.
type GeneratedVariationalDropout struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVariationalDropout() *GeneratedVariationalDropout {
	return &GeneratedVariationalDropout{Name: "variational_dropout", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVariationalDropout) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVariationalDropout) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
