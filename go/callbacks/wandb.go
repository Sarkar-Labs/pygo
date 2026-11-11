package callbacks

import "time"

// GeneratedWandb records execution state for the wandb component.
type GeneratedWandb struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWandb() *GeneratedWandb {
	return &GeneratedWandb{Name: "wandb", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWandb) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWandb) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
