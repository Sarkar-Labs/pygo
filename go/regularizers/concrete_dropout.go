package regularizers

import "time"

// GeneratedConcreteDropout records execution state for the concrete dropout component.
type GeneratedConcreteDropout struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedConcreteDropout() *GeneratedConcreteDropout {
	return &GeneratedConcreteDropout{Name: "concrete_dropout", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedConcreteDropout) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedConcreteDropout) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
