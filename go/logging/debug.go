package logging

import "time"

// GeneratedDebug records execution state for the debug component.
type GeneratedDebug struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDebug() *GeneratedDebug {
	return &GeneratedDebug{Name: "debug", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDebug) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDebug) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
