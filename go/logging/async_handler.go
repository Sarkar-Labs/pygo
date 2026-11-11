package logging

import "time"

// GeneratedAsyncHandler records execution state for the async handler component.
type GeneratedAsyncHandler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedAsyncHandler() *GeneratedAsyncHandler {
	return &GeneratedAsyncHandler{Name: "async_handler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedAsyncHandler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedAsyncHandler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
