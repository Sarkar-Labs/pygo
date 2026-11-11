package tensor

import "fmt"

func Fused0544ExpMatmulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_relu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	r = r.ReLU()
	return r, nil
}
func Fused0544ExpMatmulReluName() string { return "exp_matmul_relu" }
