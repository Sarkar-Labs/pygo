package deploy

import "time"

// GeneratedSeldon records execution state for the seldon component.
type GeneratedSeldon struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSeldon() *GeneratedSeldon {
	return &GeneratedSeldon{Name: "seldon", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSeldon) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSeldon) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
