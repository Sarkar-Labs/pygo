package tensor

import "fmt"

func Fused1354SiluMatmulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_relu: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	r = r.ReLU()
	return r, nil
}
func Fused1354SiluMatmulReluName() string { return "silu_matmul_relu" }
