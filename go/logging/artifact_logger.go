package logging

import "time"

// GeneratedArtifactLogger records execution state for the artifact logger component.
type GeneratedArtifactLogger struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedArtifactLogger() *GeneratedArtifactLogger {
	return &GeneratedArtifactLogger{Name: "artifact_logger", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedArtifactLogger) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedArtifactLogger) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
