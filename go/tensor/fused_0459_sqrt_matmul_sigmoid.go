package tensor

import "fmt"

func Fused0459SqrtMatmulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_sigmoid: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	r = r.Sigmoid()
	return r, nil
}
func Fused0459SqrtMatmulSigmoidName() string { return "sqrt_matmul_sigmoid" }
