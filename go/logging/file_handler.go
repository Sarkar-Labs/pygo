package logging

import "time"

// GeneratedFileHandler records execution state for the file handler component.
type GeneratedFileHandler struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedFileHandler() *GeneratedFileHandler {
	return &GeneratedFileHandler{Name: "file_handler", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedFileHandler) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedFileHandler) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
