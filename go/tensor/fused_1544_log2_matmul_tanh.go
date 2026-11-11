package tensor

import "fmt"

func Fused1544Log2MatmulTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_tanh: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	r = r.Tanh()
	return r, nil
}
func Fused1544Log2MatmulTanhName() string { return "log2_matmul_tanh" }
