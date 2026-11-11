package tensor

import "fmt"

func Fused0199SigmoidMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sigmoid_matmul_abs: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sigmoid()
	r = r.Abs()
	return r, nil
}
func Fused0199SigmoidMatmulAbsName() string { return "sigmoid_matmul_abs" }
