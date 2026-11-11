package callbacks

import "time"

// GeneratedProgressBar records execution state for the progress bar component.
type GeneratedProgressBar struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedProgressBar() *GeneratedProgressBar {
	return &GeneratedProgressBar{Name: "progress_bar", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedProgressBar) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedProgressBar) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
