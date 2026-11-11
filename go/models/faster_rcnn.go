package models

import "time"

// GeneratedFasterRcnn records execution state for the faster rcnn component.
type GeneratedFasterRcnn struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedFasterRcnn() *GeneratedFasterRcnn {
	return &GeneratedFasterRcnn{Name: "faster_rcnn", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedFasterRcnn) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedFasterRcnn) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
