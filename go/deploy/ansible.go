package deploy

import "time"

// GeneratedAnsible records execution state for the ansible component.
type GeneratedAnsible struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedAnsible() *GeneratedAnsible {
	return &GeneratedAnsible{Name: "ansible", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedAnsible) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedAnsible) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
