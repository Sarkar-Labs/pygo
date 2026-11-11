package logging

import "time"

// GeneratedCritical records execution state for the critical component.
type GeneratedCritical struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCritical() *GeneratedCritical {
	return &GeneratedCritical{Name: "critical", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCritical) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCritical) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
