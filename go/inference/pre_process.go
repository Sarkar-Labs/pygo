package inference

import "time"

// GeneratedPreProcess records execution state for the pre process component.
type GeneratedPreProcess struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedPreProcess() *GeneratedPreProcess {
	return &GeneratedPreProcess{Name: "pre_process", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedPreProcess) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedPreProcess) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
