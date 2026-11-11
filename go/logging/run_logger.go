package logging

import "time"

// GeneratedRunLogger records execution state for the run logger component.
type GeneratedRunLogger struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRunLogger() *GeneratedRunLogger {
	return &GeneratedRunLogger{Name: "run_logger", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRunLogger) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRunLogger) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
