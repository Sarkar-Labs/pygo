package deploy

import "time"

// GeneratedRunpod records execution state for the runpod component.
type GeneratedRunpod struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRunpod() *GeneratedRunpod {
	return &GeneratedRunpod{Name: "runpod", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRunpod) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRunpod) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
