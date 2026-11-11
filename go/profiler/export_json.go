package profiler

import "time"

// GeneratedExportJson records execution state for the export json component.
type GeneratedExportJson struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedExportJson() *GeneratedExportJson {
	return &GeneratedExportJson{Name: "export_json", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedExportJson) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedExportJson) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
