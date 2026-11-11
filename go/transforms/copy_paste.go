package transforms

import "time"

// GeneratedCopyPaste records execution state for the copy paste component.
type GeneratedCopyPaste struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCopyPaste() *GeneratedCopyPaste {
	return &GeneratedCopyPaste{Name: "copy_paste", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCopyPaste) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCopyPaste) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
