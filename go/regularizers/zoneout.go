package regularizers

import "time"

// GeneratedZoneout records execution state for the zoneout component.
type GeneratedZoneout struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedZoneout() *GeneratedZoneout {
	return &GeneratedZoneout{Name: "zoneout", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedZoneout) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedZoneout) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
