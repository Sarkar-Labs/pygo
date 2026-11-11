package training

import "time"

// GeneratedCheckpointLoad records execution state for the checkpoint load component.
type GeneratedCheckpointLoad struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCheckpointLoad() *GeneratedCheckpointLoad {
	return &GeneratedCheckpointLoad{Name: "checkpoint_load", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCheckpointLoad) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCheckpointLoad) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
