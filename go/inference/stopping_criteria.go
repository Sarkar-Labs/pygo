package inference

import "time"

// GeneratedStoppingCriteria records execution state for the stopping criteria component.
type GeneratedStoppingCriteria struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedStoppingCriteria() *GeneratedStoppingCriteria {
	return &GeneratedStoppingCriteria{Name: "stopping_criteria", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedStoppingCriteria) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedStoppingCriteria) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
