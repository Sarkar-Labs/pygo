package logging

import "time"

// GeneratedExperimentLogger records execution state for the experiment logger component.
type GeneratedExperimentLogger struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedExperimentLogger() *GeneratedExperimentLogger {
	return &GeneratedExperimentLogger{Name: "experiment_logger", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedExperimentLogger) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedExperimentLogger) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
