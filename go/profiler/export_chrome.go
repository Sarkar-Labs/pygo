package profiler

import "time"

// GeneratedExportChrome records execution state for the export chrome component.
type GeneratedExportChrome struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedExportChrome() *GeneratedExportChrome {
	return &GeneratedExportChrome{Name: "export_chrome", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedExportChrome) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedExportChrome) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
