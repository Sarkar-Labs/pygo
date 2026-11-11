package logging

import "time"

// GeneratedMetricsLogger records execution state for the metrics logger component.
type GeneratedMetricsLogger struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMetricsLogger() *GeneratedMetricsLogger {
	return &GeneratedMetricsLogger{Name: "metrics_logger", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMetricsLogger) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMetricsLogger) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
