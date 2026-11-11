package logging

import "time"

// GeneratedStreamHandler records execution state for the stream handler component.
type GeneratedStreamHandler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedStreamHandler() *GeneratedStreamHandler {
	return &GeneratedStreamHandler{Name: "stream_handler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedStreamHandler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedStreamHandler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
