package training

import (
	"encoding/gob"
	"fmt"
	"os"
)

type OptimizerState struct {
	Type                      string
	StepCount                 int64
	LR, Beta1, Beta2, Eps, WD float64
	Momentum1, Momentum2      map[string][]float64
}

func NewOptimizerState(typ string, lr, b1, b2, eps, wd float64) *OptimizerState {
	return &OptimizerState{Type: typ, LR: lr, Beta1: b1, Beta2: b2, Eps: eps, WD: wd, Momentum1: make(map[string][]float64), Momentum2: make(map[string][]float64)}
}
func (s *OptimizerState) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(s)
}
func LoadOptimizerState(path string) (*OptimizerState, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var s OptimizerState
	if err = gob.NewDecoder(f).Decode(&s); err != nil {
		return nil, err
	}
	fmt.Printf("Loaded optimizer: %s step=%d lr=%.2e\n", s.Type, s.StepCount, s.LR)
	return &s, nil
}
