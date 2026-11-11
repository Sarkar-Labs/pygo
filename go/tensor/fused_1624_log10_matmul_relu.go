package tensor

import "fmt"

func Fused1624Log10MatmulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_relu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	r = r.ReLU()
	return r, nil
}
func Fused1624Log10MatmulReluName() string { return "log10_matmul_relu" }
