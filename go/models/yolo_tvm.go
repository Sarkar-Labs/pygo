package models

import "time"

// GeneratedYoloTvm records execution state for the yolo tvm component.
type GeneratedYoloTvm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloTvm() *GeneratedYoloTvm {
	return &GeneratedYoloTvm{Name: "yolo_tvm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloTvm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloTvm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
