package logging

import "time"

// GeneratedStructuredLog records execution state for the structured log component.
type GeneratedStructuredLog struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedStructuredLog() *GeneratedStructuredLog {
	return &GeneratedStructuredLog{Name: "structured_log", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedStructuredLog) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedStructuredLog) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
