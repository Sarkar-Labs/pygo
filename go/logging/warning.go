package logging

import "time"

// GeneratedWarning records execution state for the warning component.
type GeneratedWarning struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWarning() *GeneratedWarning {
	return &GeneratedWarning{Name: "warning", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWarning) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWarning) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
