package models

import "time"

// GeneratedYoloCoreml records execution state for the yolo coreml component.
type GeneratedYoloCoreml struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloCoreml() *GeneratedYoloCoreml {
	return &GeneratedYoloCoreml{Name: "yolo_coreml", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloCoreml) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloCoreml) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
