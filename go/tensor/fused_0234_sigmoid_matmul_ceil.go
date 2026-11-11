package tensor

import "fmt"

func Fused0234SigmoidMatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_ceil: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	r = r.Ceil()
	return r, nil
}
func Fused0234SigmoidMatmulCeilName() string { return "sigmoid_matmul_ceil" }
