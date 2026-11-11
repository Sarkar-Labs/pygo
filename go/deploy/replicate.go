package deploy

import "time"

// GeneratedReplicate records execution state for the replicate component.
type GeneratedReplicate struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedReplicate() *GeneratedReplicate {
	return &GeneratedReplicate{Name: "replicate", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedReplicate) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedReplicate) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
