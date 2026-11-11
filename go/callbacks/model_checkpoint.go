package callbacks

import "time"

// GeneratedModelCheckpoint records execution state for the model checkpoint component.
type GeneratedModelCheckpoint struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedModelCheckpoint() *GeneratedModelCheckpoint {
	return &GeneratedModelCheckpoint{Name: "model_checkpoint", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedModelCheckpoint) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedModelCheckpoint) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
