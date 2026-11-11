package inference

import (
	"fmt"
	"github.com/pygo/go/tensor"
	"os"
)

type ONNXConfig struct {
	ModelPath                     string
	UseGPU                        bool
	NumThreads, OptimizationLevel int
}

func DefaultONNXConfig(path string) ONNXConfig {
	return ONNXConfig{ModelPath: path, NumThreads: 4, OptimizationLevel: 3}
}

type ONNXSession struct{ cfg ONNXConfig }

func NewONNXSession(cfg ONNXConfig) (*ONNXSession, error) {
	if _, err := os.Stat(cfg.ModelPath); err != nil {
		return nil, fmt.Errorf("not found: %s", cfg.ModelPath)
	}
	fmt.Printf("ONNX: %s\n", cfg.ModelPath)
	return &ONNXSession{cfg: cfg}, nil
}
func (s *ONNXSession) Run(inputs map[string]*tensor.Tensor) (map[string]*tensor.Tensor, error) {
	out := make(map[string]*tensor.Tensor)
	for k, v := range inputs {
		out[k] = v
	}
	return out, nil
}
func (s *ONNXSession) Close() error { return nil }
