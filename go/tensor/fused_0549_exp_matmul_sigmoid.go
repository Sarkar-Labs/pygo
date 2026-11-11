package tensor

import "fmt"

func Fused0549ExpMatmulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_sigmoid: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	r = r.Sigmoid()
	return r, nil
}
func Fused0549ExpMatmulSigmoidName() string { return "exp_matmul_sigmoid" }
