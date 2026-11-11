package serving

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

type BackendStatus int

const (
	BackendHealthy BackendStatus = iota
	BackendDegraded
	BackendDown
)

type Backend struct {
	URL              string
	Weight           int
	Status           BackendStatus
	Latency          time.Duration
	Requests, Errors atomic.Int64
	mu               sync.RWMutex
}

func (b *Backend) IsHealthy() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.Status == BackendHealthy
}

type LoadBalancer struct {
	mu       sync.RWMutex
	backends []*Backend
	strategy string
	rrIdx    uint64
}

func NewLoadBalancer(strategy string) *LoadBalancer { return &LoadBalancer{strategy: strategy} }
func (lb *LoadBalancer) AddBackend(url string, weight int) {
	lb.mu.Lock()
	defer lb.mu.Unlock()
	lb.backends = append(lb.backends, &Backend{URL: url, Weight: weight, Status: BackendHealthy})
	fmt.Printf("LB: +%s\n", url)
}
func (lb *LoadBalancer) Next() *Backend {
	lb.mu.RLock()
	defer lb.mu.RUnlock()
	var h []*Backend
	for _, b := range lb.backends {
		if b.IsHealthy() {
			h = append(h, b)
		}
	}
	if len(h) == 0 {
		return nil
	}
	switch lb.strategy {
	case "round_robin":
		return h[atomic.AddUint64(&lb.rrIdx, 1)%uint64(len(h))]
	case "least_conn":
		best := h[0]
		for _, b := range h[1:] {
			if b.Requests.Load() < best.Requests.Load() {
				best = b
			}
		}
		return best
	case "random":
		return h[rand.Intn(len(h))]
	default:
		return h[0]
	}
}
