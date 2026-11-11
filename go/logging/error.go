package logging

import "time"

// GeneratedError records execution state for the error component.
type GeneratedError struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedError() *GeneratedError {
	return &GeneratedError{Name: "error", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedError) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedError) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
