package data

import "time"

// GeneratedImageFolder records execution state for the image folder component.
type GeneratedImageFolder struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedImageFolder() *GeneratedImageFolder {
	return &GeneratedImageFolder{Name: "image_folder", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedImageFolder) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedImageFolder) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
