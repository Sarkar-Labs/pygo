package callbacks

import "time"

// GeneratedCsvLogger records execution state for the csv logger component.
type GeneratedCsvLogger struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCsvLogger() *GeneratedCsvLogger {
	return &GeneratedCsvLogger{Name: "csv_logger", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCsvLogger) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCsvLogger) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
