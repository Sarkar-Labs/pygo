package callbacks

import "time"

// GeneratedCallbackBase records execution state for the callback base component.
type GeneratedCallbackBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCallbackBase() *GeneratedCallbackBase {
	return &GeneratedCallbackBase{Name: "callback_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCallbackBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCallbackBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
