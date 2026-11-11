package tensor

import "fmt"

func Fused1634Log10MatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_tanh: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	r = r.Tanh()
	return r, nil
}
func Fused1634Log10MatmulTanhName() string { return "log10_matmul_tanh" }
