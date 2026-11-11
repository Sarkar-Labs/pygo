package logging

import "time"

// GeneratedRotatingHandler records execution state for the rotating handler component.
type GeneratedRotatingHandler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRotatingHandler() *GeneratedRotatingHandler {
	return &GeneratedRotatingHandler{Name: "rotating_handler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRotatingHandler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRotatingHandler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
