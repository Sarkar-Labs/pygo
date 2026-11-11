package tensor

import "fmt"

func Fused1359SiluMatmulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_matmul_sigmoid: nil operand")
	}
	r := x.MatMul(y)
	r = r.SiLU()
	r = r.Sigmoid()
	return r, nil
}
func Fused1359SiluMatmulSigmoidName() string { return "silu_matmul_sigmoid" }
