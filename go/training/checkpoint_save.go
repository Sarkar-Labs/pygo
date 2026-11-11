package training

import "time"

// GeneratedCheckpointSave records execution state for the checkpoint save component.
type GeneratedCheckpointSave struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCheckpointSave() *GeneratedCheckpointSave {
	return &GeneratedCheckpointSave{Name: "checkpoint_save", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCheckpointSave) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCheckpointSave) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
