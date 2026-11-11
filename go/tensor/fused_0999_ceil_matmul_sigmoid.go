package tensor

import "fmt"

func Fused0999CeilMatmulSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_sigmoid: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	r = r.Sigmoid()
	return r, nil
}
func Fused0999CeilMatmulSigmoidName() string { return "ceil_matmul_sigmoid" }
