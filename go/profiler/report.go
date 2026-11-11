package profiler

import "time"

// GeneratedReport records execution state for the report component.
type GeneratedReport struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedReport() *GeneratedReport {
	return &GeneratedReport{Name: "report", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedReport) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedReport) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
