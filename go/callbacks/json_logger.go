package callbacks

import "time"

// GeneratedJsonLogger records execution state for the json logger component.
type GeneratedJsonLogger struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedJsonLogger() *GeneratedJsonLogger {
	return &GeneratedJsonLogger{Name: "json_logger", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedJsonLogger) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedJsonLogger) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
