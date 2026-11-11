package logging

import "time"

// GeneratedJsonLog records execution state for the json log component.
type GeneratedJsonLog struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedJsonLog() *GeneratedJsonLog {
	return &GeneratedJsonLog{Name: "json_log", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedJsonLog) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedJsonLog) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
