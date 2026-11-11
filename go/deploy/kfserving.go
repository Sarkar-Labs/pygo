package deploy

import "time"

// GeneratedKfserving records execution state for the kfserving component.
type GeneratedKfserving struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedKfserving() *GeneratedKfserving {
	return &GeneratedKfserving{Name: "kfserving", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedKfserving) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedKfserving) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
