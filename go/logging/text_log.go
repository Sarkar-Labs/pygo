package logging

import "time"

// GeneratedTextLog records execution state for the text log component.
type GeneratedTextLog struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTextLog() *GeneratedTextLog {
	return &GeneratedTextLog{Name: "text_log", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTextLog) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTextLog) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
