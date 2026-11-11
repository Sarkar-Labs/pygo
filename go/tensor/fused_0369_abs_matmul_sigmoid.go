package tensor

import "fmt"

func Fused0369AbsMatmulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_sigmoid: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	r = r.Sigmoid()
	return r, nil
}
func Fused0369AbsMatmulSigmoidName() string { return "abs_matmul_sigmoid" }
