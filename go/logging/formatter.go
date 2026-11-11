package logging

import "time"

// GeneratedFormatter records execution state for the formatter component.
type GeneratedFormatter struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedFormatter() *GeneratedFormatter {
	return &GeneratedFormatter{Name: "formatter", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedFormatter) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedFormatter) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
