package profiler

import (
	"fmt"
	"sort"
	"sync"
	"time"
)

type ProfileEvent struct {
	Name       string
	Start, End time.Time
}

func (e *ProfileEvent) Duration() time.Duration { return e.End.Sub(e.Start) }

type Profiler struct {
	mu      sync.Mutex
	events  []*ProfileEvent
	active  map[string]time.Time
	enabled bool
}

var Global = &Profiler{enabled: true, active: make(map[string]time.Time)}

func (p *Profiler) Start(name string) {
	if !p.enabled {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.active[name] = time.Now()
}
func (p *Profiler) End(name string) time.Duration {
	if !p.enabled {
		return 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	s, ok := p.active[name]
	if !ok {
		return 0
	}
	d := time.Since(s)
	delete(p.active, name)
	p.events = append(p.events, &ProfileEvent{Name: name, Start: s, End: s.Add(d)})
	return d
}
func (p *Profiler) Time(name string, fn func()) time.Duration {
	p.Start(name)
	fn()
	return p.End(name)
}
func (p *Profiler) Summary() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	tots := map[string]time.Duration{}
	cnts := map[string]int{}
	for _, e := range p.events {
		tots[e.Name] += e.Duration()
		cnts[e.Name]++
	}
	type row struct {
		n string
		t time.Duration
		c int
	}
	var rows []row
	for n, t := range tots {
		rows = append(rows, row{n, t, cnts[n]})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].t > rows[j].t })
	out := "=== PyGo Profiler ===\n"
	for _, r := range rows {
		out += fmt.Sprintf("  %-40s total=%-10s avg=%-10s n=%d\n", r.n, r.t.Round(time.Microsecond), (r.t / time.Duration(r.c)).Round(time.Microsecond), r.c)
	}
	return out
}
func (p *Profiler) Reset() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.events = nil
	p.active = make(map[string]time.Time)
}
