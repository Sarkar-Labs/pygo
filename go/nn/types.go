package nn

import "sync"

// GeneratedTypesState is a small concurrency-safe state container used by types.
type GeneratedTypesState struct {
	mu     sync.RWMutex
	values map[string]float64
}

// NewGeneratedTypesState creates an initialized state object.
func NewGeneratedTypesState() *GeneratedTypesState {
	return &GeneratedTypesState{values: make(map[string]float64)}
}

func (s *GeneratedTypesState) Set(key string, value float64) {
	s.mu.Lock()
	s.values[key] = value
	s.mu.Unlock()
}
func (s *GeneratedTypesState) Get(key string) (float64, bool) {
	s.mu.RLock()
	v, ok := s.values[key]
	s.mu.RUnlock()
	return v, ok
}
