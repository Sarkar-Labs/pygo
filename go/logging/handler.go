package logging

import "time"

// GeneratedHandler records execution state for the handler component.
type GeneratedHandler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedHandler() *GeneratedHandler {
	return &GeneratedHandler{Name: "handler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedHandler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedHandler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
